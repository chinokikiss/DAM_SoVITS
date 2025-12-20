import os
from typing import Dict, List
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def batch_sequences(sequences: List[np.array], axis: int = 0, pad_value: int = 0):
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_value = dtype.type(pad_value)
    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = np.max(seq_lengths)

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = [(0, 0)] * axis + [(0, max_length - length)] + [(0, 0)] * (ndim - axis - 1)
        padded_seq = np.pad(seq, padding, mode="constant", constant_values=pad_value)
        padded_sequences.append(padded_seq)
    batch = np.stack(padded_sequences)
    return batch


class Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(
        self,
        data_path: str,
        max_sec: int = 100,
        # min value of phoneme/sec
        min_ps_ratio: int = 3,
        # max value of phoneme/sec
        max_ps_ratio: int = 25,
    ) -> None:
        super().__init__()

        data_file = np.load(data_path, allow_pickle=True).item()
        self.data_duration, self.data = data_file["data_duration"], data_file["datas"]
        self.speaker_idx = None

        self.hz = int(os.environ.get("hz", "25hz")[:-2])

        # max seconds of semantic token
        self.max_sec = max_sec
        self.min_ps_ratio = min_ps_ratio
        self.max_ps_ratio = max_ps_ratio

        self.inited = False

        self.EOS = 1024
        self.ClS = 1026
        self.SEP = 512

        if not self.inited:
            # 调用初始化函数
            self.init_batch()
            self.inited = True

    def init_batch(self):
        data = []
        self.speaker_idx = {}
        idx = 0
        num_not_in = 0
        num_deleted_bigger = 0
        num_deleted_ps = 0

        for speaker, semantic_ids, phoneme_ids, emo_feature in self.data:
            # 删除语义过长的数据
            if len(semantic_ids) > self.max_sec * self.hz:
                num_deleted_bigger += 1
                continue
            
            # 删除音素过长的数据
            if len(phoneme_ids) > self.max_sec * self.hz / 2.5:
                num_deleted_ps += 1
                continue
            
            # 删除音素-语义比率过大的数据
            ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self.hz)
            if ps_ratio > self.max_ps_ratio or ps_ratio < self.min_ps_ratio:
                num_deleted_ps += 1
                continue
            
            if speaker not in self.speaker_idx:
                self.speaker_idx[speaker] = []
            self.speaker_idx[speaker].append(idx)
            data.append((speaker, semantic_ids, np.array(phoneme_ids), emo_feature))
            idx += 1

        if num_not_in > 0:
            print(f"there are {num_not_in} semantic datas not in phoneme datas")
        if num_deleted_bigger > 0:
            print(
                f"deleted {num_deleted_bigger} audios who's duration are bigger than {self.max_sec} seconds",
            )
        if num_deleted_ps > 0:
            print(
                f"deleted {num_deleted_ps} audios who's phoneme/sec are bigger than {self.max_ps_ratio} or smaller than {self.min_ps_ratio}",
            )
        
        self.data = data

        print()
        print("数据量:", self.__len__())
        print(f"数据总时长: {int(self.data_duration/3600)}h or {int(self.data_duration/60)}m")
        print("说话人数量:", len(self.speaker_idx))
        print()

    def __len__(self) -> int:
        return len(self.data)

    def get_sample_length(self, idx: int) -> int:
        _, semantic_ids, _, _ = self.data[idx]
        return len(semantic_ids)

    def __getitem__(self, idx: int) -> Dict:
        speaker, semantic_ids, phoneme_ids, emo_feature = self.data[idx]

        return {
            "speaker": speaker,
            "phoneme_ids": phoneme_ids,
            "semantic_ids": semantic_ids,
        }

    def collate(self, examples: List[Dict]) -> Dict:
        phoneme_ids: List[torch.Tensor] = []
        refer_phoneme_ids_len: List[int] = []
        target_phoneme_ids_len: List[int] = []
        semantic_ids: List[torch.Tensor] = []
        refer_semantic_ids_len: List[int] = []
        target_semantic_ids_len: List[int] = []
        emo_feature: List[torch.Tensor] = []

        for item in examples:
            target_idx = random.choice(self.speaker_idx[item["speaker"]])
            _, _semantic_ids, _phoneme_ids, _emo_feature = self.data[target_idx]

            phoneme_ids.append(np.concatenate([item["phoneme_ids"], [self.SEP], _phoneme_ids]))
            refer_phoneme_ids_len.append(len(item["phoneme_ids"]))
            target_phoneme_ids_len.append(len(_phoneme_ids))

            _semantic_ids = np.concatenate([_semantic_ids, [self.EOS]])
            semantic_ids.append(np.concatenate([item["semantic_ids"], [self.ClS], _semantic_ids]))
            refer_semantic_ids_len.append(len(item["semantic_ids"]))
            target_semantic_ids_len.append(len(_semantic_ids))

            emo_feature.append(_emo_feature)

        phoneme_ids = batch_sequences(phoneme_ids)
        semantic_ids = batch_sequences(semantic_ids, pad_value=self.EOS)

        phoneme_ids = torch.tensor(phoneme_ids)
        refer_phoneme_ids_len = torch.tensor(refer_phoneme_ids_len)
        target_phoneme_ids_len = torch.tensor(target_phoneme_ids_len)
        semantic_ids = torch.tensor(semantic_ids)
        refer_semantic_ids_len = torch.tensor(refer_semantic_ids_len)
        target_semantic_ids_len = torch.tensor(target_semantic_ids_len)
        emo_feature = torch.from_numpy(np.array(emo_feature))

        return {
            "phoneme_ids": phoneme_ids,
            "refer_phoneme_ids_len": refer_phoneme_ids_len,
            "target_phoneme_ids_len": target_phoneme_ids_len,
            "semantic_ids": semantic_ids,
            "refer_semantic_ids_len": refer_semantic_ids_len,
            "target_semantic_ids_len": target_semantic_ids_len,
            "emo_feature": emo_feature,
        }


if __name__ == "__main__":
    dataset = Text2SemanticDataset(
        data_path=r"C:\Users\23709\Documents\new_work\柯莱.npy"
    )

    batch_size = 12
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate,
        shuffle=False,
    )
    
    for i, batch in enumerate(dataloader):
        if i % 100 == 0:
            print(batch)
            input()