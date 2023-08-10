from collections import defaultdict

from .base_dataset import BaseDataset


class MIMICCXRDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["mimic_cxr_train"]
        elif split == "val":
            names = ["mimic_cxr_val"]
        elif split == "test":
            names = ["mimic_cxr_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        self.chexpert_labels = self.table["chexpert"].to_pandas().tolist()
        dup_indices = [self.index_mapper[i][0] for i in self.index_mapper]
        self.chexpert_labels = [self.chexpert_labels[idx] for idx in dup_indices]
        self.group_mappings = defaultdict(set)
        for idx, label in enumerate(self.chexpert_labels):
            self.group_mappings[str(label)].add(idx)
        full_index_set = set(list(range(len(self.chexpert_labels))))
        for k in self.group_mappings:
            self.group_mappings[k] = full_index_set - self.group_mappings[k]

    # def get_false_image(self, rep, image_key="image", selected_index=None):
    #     chexpert_label = str(self.chexpert_labels[selected_index])
    #     candidate_index = self.group_mappings[chexpert_label]
    #
    #     random_index = random.sample(candidate_index, 1)[0]
    #     image = self.get_raw_image(random_index, image_key=image_key)
    #     image_tensor = [tr(image) for tr in self.transforms]
    #     return {f"false_image_{rep}": image_tensor}
    #
    # def get_false_text(self, rep, selected_index=None):
    #     chexpert_label = str(self.chexpert_labels[selected_index])
    #     candidate_index = self.group_mappings[chexpert_label]
    #
    #     random_index = random.sample(candidate_index, 1)[0]
    #     index, caption_index = self.index_mapper[random_index]
    #     text = self.all_texts[index][caption_index]
    #     encoding = self.tokenizer(
    #         text,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_text_len,
    #         return_special_tokens_mask=True,
    #         return_offsets_mapping=True,
    #     )
    #     return {f"false_text_{rep}": (text, encoding)}

    def __getitem__(self, index):
        return self.get_suite(index)
