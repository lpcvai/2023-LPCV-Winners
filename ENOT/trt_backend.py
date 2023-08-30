from typing import Dict, NamedTuple
from pathlib import Path

import torch
import tensorrt

TRT_LOGGER = tensorrt.Logger(tensorrt.Logger.ERROR)
builder = tensorrt.Builder(TRT_LOGGER)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TensorrtBackend:
    class Binding(NamedTuple):
        name: str
        tensor: torch.Tensor
        ptr: int

    _INPUT_NAME = "input"
    _OUTPUT_1_NAME = "output"

    def __init__(self, engine_file: Path):
        self._engine = self._init_engine(engine_file)
        self._context = self._engine.create_execution_context()

        assert self._engine.num_bindings == 2  # one input, one output
        assert self._engine.get_binding_name(0) == TensorrtBackend._INPUT_NAME
        assert self._engine.get_binding_name(1) == TensorrtBackend._OUTPUT_1_NAME

        self._bindings: Dict[str, TensorrtBackend.Binding] = {}

        # input binding:
        self._bindings[TensorrtBackend._INPUT_NAME] = TensorrtBackend.Binding(
            name=TensorrtBackend._INPUT_NAME,
            tensor=None,  # type: ignore
            ptr=0,
        )

        # allocate memory for outputs:
        name = self._engine.get_binding_name(1)
        shape = tuple(self._engine.get_binding_shape(1))
        tensor = torch.empty(shape, dtype=torch.float32, device="cuda")
        self._bindings[name] = TensorrtBackend.Binding(
            name=name,
            tensor=tensor,
            ptr=int(tensor.contiguous().data_ptr()),
        )

        self._binding_addrs: Dict[str, int] = {
            TensorrtBackend._INPUT_NAME: self._bindings[TensorrtBackend._INPUT_NAME].ptr,
            TensorrtBackend._OUTPUT_1_NAME: self._bindings[TensorrtBackend._OUTPUT_1_NAME].ptr,
        }

    def infer(self, inputs: torch.Tensor) -> torch.Tensor:
        self._binding_addrs[TensorrtBackend._INPUT_NAME] = inputs.contiguous().data_ptr()

        is_success = self._context.execute_v2(bindings=[*self._binding_addrs.values()])
        if not is_success:
            raise RuntimeError("TensorRT execution unexpectedly failed")

        return self._bindings[TensorrtBackend._OUTPUT_1_NAME].tensor,

    def _init_engine(self, engine_file: Path) -> tensorrt.ICudaEngine:
        runtime = tensorrt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_file.read())
        return engine

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.infer(inputs)
