

from datasets import DatasetInfo, Features, Split, SplitGenerator, GeneratorBasedBuilder, Value
import json

class MyDataset(GeneratorBasedBuilder):
    def _info(self):
        return DatasetInfo(
            features=Features({
                "id": Value("int32"),
                "question": Value("string"),
                "answer": Value("string"),
                "score": Value("int16"),
                "label": Value("string"),
            }),
            supervised_keys=("question","answer"),
            homepage="https://github.com/FreedomIntelligence/Huatuo-26M",
            citation='''
                        @misc{li2023huatuo26m,
                              title={Huatuo-26M, a Large-scale Chinese Medical QA Dataset}, 
                              author={Jianquan Li and Xidong Wang and Xiangbo Wu and Zhiyi Zhang and Xiaolong Xu and Jie Fu and Prayag Tiwari and Xiang Wan and Benyou Wang},
                              year={2023},
                              eprint={2305.01526},
                              archivePrefix={arXiv},
                              primaryClass={cs.CL}
                        }

                        ''',
        )

    def _split_generators(self, dl_manager):

        test_path = "format_data.jsonl"

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                # Process your data here and create a dictionary with the features.
                # For example, if your data is in JSON format:
                data = json.loads(row)
                yield id_, {
                    "id": data["id"],
                    "question": data["question"],
                    "answer": data["answer"],
                    "label": data["label"],
                    "score": data["score"]
                }

if __name__ == '__main__':
    from datasets import load_dataset

    dataset = load_dataset("my_dataset.py")

    print()