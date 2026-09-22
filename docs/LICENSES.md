# Licences and attribution

These are the component terms for this public experimental release. The project is an adaptation of Qwen using a dataset with Google and original TOPv2 notices; **the complete package must not be presented as Apache-only**.

| Component | Release terms |
| --- | --- |
| Udgam's trained adapter contribution | CC BY-SA 4.0 as a conservative distribution choice |
| Prepared Hinglish-TOP and retrieval examples | CC BY-SA 4.0, retaining Google and TOPv2 notices |
| Recorded dataset examples and adapted-data excerpts | Retain dataset attribution and CC BY-SA 4.0 lineage |
| Udgam-written training, evaluation, wrapper and demo code | Apache 2.0 for Udgam's licensable contribution |
| Original Qwen files, tokenizer and unchanged third-party materials | Preserve their existing terms and notices; no exclusive ownership or relicensing claim |
| Third-party dependencies | Their respective licences |

CC BY-SA 4.0 permits reuse, including commercial reuse, subject to its conditions. Preserve attribution, link the licence, indicate changes and follow its share-alike terms where applicable. Do not add a noncommercial restriction. The choice for the adapter is conservative; it is not a conclusion that every AI training operation necessarily creates a copyright adaptation. [Creative Commons AI-training guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) discusses that uncertainty. The [CC BY-SA 4.0 licence](https://creativecommons.org/licenses/by-sa/4.0/) and its linked legal code govern, rather than this short summary.

No rights are asserted over facts, public-domain material or a user's original command just because the parser processes it. Different terms apply to original upstream materials and Udgam's contributions.

## Credits

Base model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), revision `c1899de289a04d12100db370d81485cdf75e47ca`. Dataset: [Google Research Hinglish-TOP](https://github.com/google-research-datasets/Hinglish-TOP-Dataset/tree/fdd3998a6573130659bfa1ce4b1ebe698df2bf3a), revision `fdd3998a6573130659bfa1ce4b1ebe698df2bf3a`, building on [TOPv2](https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip). The original authors have not endorsed this adaptation.

Hinglish-TOP accompanies **CST5: Data Augmentation for Code-Switched Semantic Parsing**, Anmol Agarwal, Jigar Gupta, Rahul Goel, Shyam Upadhyay, Pankaj Joshi and Rengarajan Aravamudhan (2023). [Paper](https://aclanthology.org/2023.tllm-1.1/). Preserve the Google repository's Apache notice and README in addition to the underlying dataset notice.

TOPv2 accompanies **Low-Resource Domain Adaptation for Compositional Task-Oriented Semantic Parsing**, Xilun Chen, Asish Ghoshal, Yashar Mehdad, Luke Zettlemoyer and Sonal Gupta (2020). [Paper](https://aclanthology.org/2020.emnlp-main.413/). Its original archive contains CC BY-SA 4.0 terms.

The measured comparator was [KingNish/Qwen3-0.6b-hinglish-2](https://huggingface.co/KingNish/Qwen3-0.6b-hinglish-2/tree/fecc11f386ae5d30f84cfd8bb90b1512b5ba6188), revision `fecc11f386ae5d30f84cfd8bb90b1512b5ba6188`. It was evaluated as a baseline; its weights are not included in this release.

## Changes made by Udgam

Udgam removed detected training/validation overlap and duplicates, excluded documented unusable/conflicting pairs, emitted paired English/Hinglish records, adapted Qwen with the selected human-only LoRA recipe, evaluated against existing labels, and built a deterministic TOP-to-JSON preview interface. Utterance and target wording was retained; normalization was used for overlap detection and separately documented evaluation procedures. The final benchmark was preserved instead of being rewritten to make predictions look better. The optional synthetic-training variant was rejected and is not part of the selected adapter.

## Notices to keep with the release

- Qwen's original Apache 2.0 licence and attribution.
- Google Hinglish-TOP Apache notice and source README.
- Original TOPv2 CC BY-SA 4.0 notice and source README.
- Udgam's applicable code/adapter notices and a description of changes.
- Pinned revisions and source hashes, including the retrieval corpus.

An Apache code licence at repository root does not override the data and adapter terms. Packaging metadata should describe the adapter as `cc-by-sa-4.0`, and documentation should explain this component split.
