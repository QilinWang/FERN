import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import einops   
from typing import (
    List, Tuple, Union, Callable, Dict, Optional, Any, Self, Literal,
    Annotated, Iterable,  
    Set,NamedTuple, Callable 
) 
import study.metrics as metrics 
import study.FERN_util as fru 
import study.configs as configs 
from pydantic import (
    BaseModel, PositiveInt, ConfigDict, Field, computed_field, model_validator
) 
from enum import StrEnum, auto
import study.FERN_core as fcore 
import study.FERN_gen as fgen
import study.FERN_apply as fapply

  
class PipelinePhase:
    def __init__(self, flows: dict, sequence: list[str]):
        self.flows = flows
        self.sequence = sequence
    
    def apply(self, states: fgen.States) -> fgen.States:
        for name in self.sequence:
            if name:
                states = states | self.flows[name]
        return states
    
    def __repr__(self):
        return f"{self.__class__.__name__}({' → '.join(self.sequence)})"

class EncodingPhase(PipelinePhase): pass
class ProcessingPhase(PipelinePhase): pass  
class DecodingPhase(PipelinePhase): pass


class FERN(nn.Module):
    """
    """
    def __init__(self, cfg: configs.FERNConfig): 
        super().__init__() 
        self.cfg = cfg 

        self.fixed_params = nn.ParameterDict({
            "real_scale": nn.Parameter(torch.ones(cfg.channels, cfg.dim_augment)),
            "complex_a":  nn.Parameter(torch.ones(cfg.channels, cfg.dim_augment // 2) * 1.0),
            "complex_b":  nn.Parameter(torch.zeros(cfg.channels, cfg.dim_augment // 2) + 0.0),
            "complex_a_y":  nn.Parameter(torch.ones(cfg.channels, cfg.pred_len // 2) * 1.0),
            "complex_b_y":  nn.Parameter(torch.zeros(cfg.channels, cfg.pred_len // 2) + 0.0),
        })
        self.k = fgen.Koopman(a=self.fixed_params['complex_a'], b=self.fixed_params['complex_b'])
        self.ky = fgen.Koopman(a=self.fixed_params['complex_a_y'], b=self.fixed_params['complex_b_y'])
        self.moving_avg = fru.moving_avg(kernel_size=25, stride=1) 
        self.NNs = fgen.build_factories(cfg) 
        self.conv = nn.Conv1d(self.cfg.channels, self.cfg.channels, kernel_size=1, groups=self.cfg.channels)
        
        # Probabilistic source encoding 
        self.prob_g_hid_from_x = fapply.ProbG(choices=[
            (fapply.G(field="x", fac=self.NNs["HID_GIVEN_X_V0"], augment='none'), 0.4),
            (fapply.G(field="x", fac=self.NNs["HID_GIVEN_X_V0"], augment='boot_odd_keep_even'), 0.3), # boot_odd_keep_even
            (fapply.G(field="x", fac=self.NNs["HID_GIVEN_X_V0"], augment='even'), 0.3), # even
        ]) 
        # === SEMANTIC PIPELINE PHASES === 
        self.phases = self._build_pipeline_phases() 
        self.a = nn.Parameter(torch.ones(cfg.channels, cfg.pred_len))
        self.f = nn.Parameter(torch.ones(1)*0.5)
    def _build_pipeline_phases(self) -> dict:
        """Build semantically meaningful pipeline phases"""
        
        def bidirectional_flow(src: str, dst: str, src_geom: fapply.Geom,  versions: list[str], dst_geom: fapply.Geom = None):
            """Create bidirectional flows between two spaces"""
            if dst_geom is None:
                dst_geom = src_geom
            return {
                f"{src.upper()}_to_{dst.upper()}_{v.upper()}": fapply.compile_chain(
                    fapply.RECIPES[src_geom], src=src, dst=dst, fac=self.NNs, version=v)
                for v in versions
            } | {
                f"{dst.upper()}_to_{src.upper()}_{v.upper()}": fapply.compile_chain(
                    fapply.RECIPES[dst_geom], src=dst, dst=src, fac=self.NNs, version=v) 
                for v in versions
            }
        
        flows = {}
        
        # === ENCODING PHASE: X ↔ Z (latent space) ===
        flows.update(bidirectional_flow(
            "x", "z", fapply.Geom.SCALE_SHIFT, ["v2","v3","v4","v5"], fapply.Geom.SCALE_SHIFT)) # PROB_SCALE_SHIFT SCALE_SHIFT
        
        # Special probabilistic X→Z
        flows["X_to_Z_PROB"] = (fapply.I() | self.prob_g_hid_from_x 
                           | fapply.G(field="h_re", fac=self.NNs["SS_Z_GIVEN_X_V0"])
                           | fapply.T(field="z", op_name=fapply.TOp.SCALE) 
                           | fapply.T(field="z", op_name=fapply.TOp.SHIFT))
        
        flows["Z_to_X_V0"] = fapply.compile_chain(
            fapply.RECIPES[fapply.Geom.SCALE_SHIFT], src="z", dst="x", fac=self.NNs, version="v0")
        flows["X_to_Z_V0"] = fapply.compile_chain(
            fapply.RECIPES[fapply.Geom.SCALE_SHIFT], src="x", dst="z", fac=self.NNs, version="v0")
        flows['KOOPMAN'] = fapply.compile_chain(
            fapply.RECIPES[fapply.Geom.KOOP], src="z", dst="z", fac=self.NNs, version="v0")
        # === PROCESSING PHASE: Y ↔ Z (prediction-latent interaction) ===
        flows.update(bidirectional_flow(
            "y", "z", fapply.Geom.SCALE_SHIFT, ["v2","v3","v4"], fapply.Geom.SHIFT_ONLY))
        
        # === DYNAMICAL PHASE: Z → Z (Koopman evolution) ===
        flows["Z_to_Z_K"] = fapply.compile_chain(
            fapply.RECIPES[fapply.Geom.R_KOOP_RB_SHIFT], src="z", dst="z", fac=self.NNs, version="v0")
        # flows["Z_to_Z_DIRECT"] = fapply.compile_chain(fapply.RECIPES[fapply.Geom.SCALE_SHIFT], src="z", dst="z", fac=self.NNs)
        
        # === DECODING PHASE: Z → Y (output generation) ===
        flows["Z_to_Y_OT"] = fapply.compile_chain(
            fapply.RECIPES[fapply.Geom.R_SCALE_RB_SHIFT], src="z", dst="y", fac=self.NNs, version="v0")
        # flows["Z_to_Y_CASCADE"] = fapply.compile_chain(
        #     fapply.RECIPES[fapply.Geom.CASCADE], src="z", dst="y", fac=self.NNs, version="v0")
        
         
        return {
            "encoding": EncodingPhase(flows, [
                "X_to_Z_PROB", 
                # "X_to_Z_V0",
                "Z_to_X_V0", 
                "X_to_Z_V2", 
                # "Z_to_Y_V2",
                # "Y_to_Z_V2",
                "Z_to_X_V2", 
                "X_to_Z_V3", 
                # "Z_to_Y_V3",
                # "Y_to_Z_V3",
                "Z_to_X_V3", 
                "X_to_Z_V4", 
                "Z_to_Y_V4",
                # "Y_to_Z_V4",
                # "Z_to_X_V4", 
                "X_to_Z_V5", 
                "Z_to_X_V5"
            ]),
            "processing": ProcessingPhase(flows, [ 
                "Z_to_Y_V2", 
                "Y_to_Z_V2", 
                # "KOOPMAN",
                "Z_to_Y_V3",  
                "Y_to_Z_V3", 
                "Z_to_Y_V4", 
                "Y_to_Z_V4", 
                
            ]),
            "decoding": DecodingPhase(flows, [
                "Z_to_Y_OT",
                # "Z_to_Y_CASCADE",
                # "Y_to_Z_V4", "Z_to_Y_V4", 
            ]),
            "flows": flows  # raw access if needed
        }
    
    def forward(self, x_bsd: torch.Tensor, update: bool = True) -> Dict[str, torch.Tensor]:
        """
        Chain example : plan = Chain([ rotate, op, ~rotate, params_for_z_given_z.shift] )
        states.z = states.z | plan
        """
        x = x_bsd.permute(0, 2, 1) 
        y_shape = fru.dynamic_size(x, self.cfg.pred_len)
        z_shape = fru.dynamic_size(x, self.cfg.dim_augment) 
        z = fru.sample_base(z_shape, x.device, x.dtype, kind="gauss")*0.1
        y = fru.sample_base(y_shape, x.device, x.dtype, kind="gauss")* 0.1
        # y = y + self.a
        # y = y * self.param_scale_y + self.param_shift_y
        # states = states.set_y(y)  
  
        with torch.set_grad_enabled(update):  
            # x = x * self.param_scale_x + self.param_shift_x
            # z = z * self.param_scale_z + self.param_shift_z
            # x = self.conv(x)
            
            # Initialize state
            states = fgen.States(
                x=x.clone(), 
                z=z, 
                y=y, 
                koopman=self.k, 
                is_training=self.training, 
                resid_factor=self.f
                )
            # states2 = fgen.States(
            #     x=x.clone(), 
            #     z=fru.sample_base(z_shape, x.device, x.dtype, kind="gumbel")*0.1, 
            #     y=fru.sample_base(y_shape, x.device, x.dtype, kind="gumbel")*0.1, 
            #     koopman=self.k, 
            #     is_training=self.training, 
            #     resid_factor=self.f
            #     )

            # === PHASE 1: ENCODE X → Z ===
            
            states = self.phases["encoding"].apply(states)  #TODO
            
            # states2 = self.phases["encoding"].apply(states2)
            
            
            # === PHASE 2: INITIALIZE Y === 
            
            
            # === PHASE 3: PROCESS Y ↔ Z ===
            
            # y = self.a(states.z) #+ states.y
            # states = states.set_y(y)
            
            # states = self.phases["processing"].apply(states) #TODO
            
            # states2 = self.phases["processing"].apply(states2)
            pre_y =  states.y
            
            # === PHASE 4: DECODE Z → Y ===
            # λ = 0.2
            # ctx = (1-λ) * z.detach() + λ * z
            # states = states.set_z(ctx)
            # states = self.phases["decoding"].apply(states) 
            # states2 = self.phases["decoding"].apply(states2)
            # cascade
            rotation = self.NNs["ROT_IN_Y_V1"](states.z)
            scale, shift = self.NNs["KOO_Y_GIVEN_Z_V0"](states.z)
            states.rotation = rotation
            # print(scale.coef)
            constant = pre_y | rotation  | self.ky
            # print(self.ky.a)
            accu = constant | scale
            # accu = (constant + torch.sigmoid(accu)) #| scale
            accu = accu |  self.ky.inv | rotation.inv | shift # .inv
            states.y = accu #* self.f + states.y
            
            
            # states.y = states.y * 0.5 + states2.y * 0.5
        return states.y.permute(0, 2, 1)
