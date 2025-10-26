from abc import ABC, abstractmethod
from typing import Any, NotRequired, Required, TypedDict, cast

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from numpy.typing import NDArray
from streaming import StreamingDataset
from streaming.base import MDSWriter
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


class ContextEmbedderOutput(TypedDict):
    embeddings: Required[Tensor]
    pooled_embeddings: NotRequired[Tensor]
    attn_mask: NotRequired[Tensor]


class ContextEmbedder(nn.Module, ABC):
    _n_output: int
    _output_size: tuple[int, ...]

    def __init__(self):
        super().__init__()  # type: ignore

    @property
    def n_output(self) -> int:
        """
        Represents the number of output embedding the embedder is returning.
        """
        return self._n_output

    @property
    def output_size(self) -> tuple[int, ...]:
        """
        Represents the dimension of each output embedding.
        """
        return self._output_size

    @abstractmethod
    def drop_conditions(
        self,
        context: Any,
        p: float,
    ) -> Any:
        """
        Randomly drop drop_context from a batch.

        Args:
            drop_context (Any): a sequence of context.
            p (float): the probability of dropping a context.
        Returns
            Any: the same sequence of context with some elements randomly dropped.
        """
        pass

    @abstractmethod
    def forward(self, context: Any, p: float = 0) -> ContextEmbedderOutput:
        """
        Apply the model to an input batch.

        Args:
            context (Any): the input batch, can be a tensor or a list of str for example.
            p (float): the probability of dropping the context.
        Returns:
            ContextEmbedderOutput: a dictionary containing the output embeddings and
            optionally pooled embeddings and mask.
        """
        pass

    @torch.inference_mode()
    def compute_on_dataset(
        self,
        dataset_path: str,
        dst_path: str | tuple[str, str],
        local: bool = True,
        batch_size: int = 64,
        split: str | None = None,
        to_process_data_key: str | None = None,
        target_type: str = "float32",  # "float32" | "float16"
        column_prefix: str = "context_embeddings",
    ) -> None:
        """
        Run the embedder over a MosaicML StreamingDataset and write an MDS with
        the original columns + one ndarray column per output of the embedder.

        Each output i is written to a column named f"{column_prefix}_{i}" with
        encoding "ndarray:<target_type>".

        Args:
            dataset_path: Path to existing StreamingDataset (local dir or remote URI).
            dst_path: Destination for MDSWriter (dir or (remote, local) tuple).
            local: If True, treat dataset_path as local; otherwise as remote.
            batch_size: Loader batch size (controls tokenization/forward work size).
            split: Optional split name.
            to_process_data_key: Explicit column name containing data to process. If None,
            target_type: "float32" or "float16" for output arrays.
            column_prefix: Prefix for destination embedding columns.
            progress_desc: TQDM description.
        """
        dataset = StreamingDataset(
            remote=dataset_path if not local else None,
            local=dataset_path if local else None,
            batch_size=batch_size,
            split=split,
        )
        assert dataset.shards, "Dataset has no shards."  # type: ignore

        def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            return {key: [item[key] for item in batch] for key in batch[0].keys()}

        dataloader = cast(
            DataLoader[dict[str, Any]],
            DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn),
        )

        source_columns: dict[str, str] = {
            name: encoding
            for name, encoding in zip(dataset.shards[0].column_names, dataset.shards[0].column_encodings)  # type: ignore
        }

        dst_columns = dict(source_columns)
        for i in range(self.n_output):
            dst_columns[f"{column_prefix}_{i}"] = f"ndarray:{target_type}"

        # device / dtype (best effort; works even if submodules are off-device)
        try:
            param = next(self.parameters())
            device = param.device
            dtype = param.dtype
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            logger.warning(f"device and dtype not found | defaulting to {device} and {dtype}.")

        # Main loop
        with MDSWriter(out=dst_path, columns=dst_columns) as writer:
            for batch in tqdm(dataloader, desc="computing embeddings", unit="batch"):
                to_process_data = batch[to_process_data_key]

                # Forward -> tuple of Tensors
                outputs = self.forward(to_process_data)

                # Move to CPU, cast dtype, convert to numpy
                np_outputs: dict[str, NDArray[np.floating[Any]]] = {}
                for key, out in outputs.items():
                    assert isinstance(out, Tensor), "Only Tensor outputs are supported."
                    out = out.detach().to("cpu").float()
                    arr = out.numpy()  # type: ignore[reportUnknownMemberType]
                    if target_type == "float32":
                        arr = arr.astype(np.float32)
                    elif target_type == "float16":
                        arr = arr.astype(np.float16)
                    else:
                        raise ValueError("target_type must be 'float32' or 'float16'")
                    np_outputs[key] = arr

                B = len(to_process_data)
                # Write per-sample rows, preserving original columns
                for i in range(B):
                    sample = {k: v[i] for k, v in batch.items()}
                    for key, arr in np_outputs.items():
                        sample[f"{column_prefix}_{key}"] = arr[i]
                    writer.write(sample)
