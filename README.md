# vaik-text-recognition-trt-experiment

Create json file by text recognition model. Calc Levenshtein ratio.

## Install

```shell
pip install -r requirements.txt
```

## Docker Install

- amd64(g4dn.xlarge)

```shell
docker build -t g4dnxl_ed_experiment -f ./Dockerfile.g4dn.xlarge .
sudo docker run --runtime=nvidia \
           --name g4dnxl_ed_experiment_container \
           --rm \
           -v ~/.vaik_text_recognition_pb_trainer/dump_dataset:/workspace/dump_dataset \
           -v ~/output_trt_model:/workspace/output_trt_model \
           -v $(pwd):/workspace/source \
           -it g4dnxl_ed_experiment /bin/bash
```

- arm64(JetsonXavierNX)

```shell
sudo docker build -t jxnj502_experiment -f ./Dockerfile.jetson_xavier_nx_jp_502 .
sudo docker run --runtime=nvidia \
           --name jxnj502_experiment_container \
           --rm \
           -v ~/.vaik_text_recognition_pb_trainer/dump_dataset:/workspace/dump_dataset \
           -v ~/output_trt_model:/workspace/output_trt_model \
           -v $(pwd):/workspace/source \
           -it jxnj502_experiment /bin/bash
```

---------

## Usage

### Create json file

```shell
python3 inference.py --input_saved_model_file_path '/workspace/output_trt_model/model.trt' \
                --input_classes_json_path '/workspace/source/test_default_fonts_images/jpn_character.json' \
                --input_image_dir_path '/workspace/source/test_default_fonts_images' \
                --output_json_dir_path '/workspace/output_tflite_model/test_default_fonts_images_out'
```

- input_image_dir_path
    - example

```shell
.
├── なにわ_3932.png
├── 京都_0656.png
├── 倉敷_0488.png
・・・
```

#### Output
- output_json_dir_path
    - example

```json
[{
  "answer": "いわき",
  "classes": [
    113,
    155,
    118
  ],
  "image_path": "/workspace/source/test_default_fonts_images/いわき_00122.jpg",
  "scores": 1.1269535508079428e-23,
  "text": "いわき"
}
]
```
-----

### Calc Levenshtein Ratio

```shell
python calc_levenshtein_ratio.py --input_json_dir_path '~/output_tflite_model/test_default_fonts_images_inference'
```

#### Output

``` text
ratio:1.0, answer:佐世保, predict:佐世保
ratio:1.0, answer:とちぎ, predict:とちぎ
ratio:1.0, answer:いわき, predict:いわき
ratio:1.0, answer:大分, predict:大分
ratio:1.0, answer:三重, predict:三重
ratio:1.0, answer:八王子, predict:八王子
ratio:1.0, answer:名古屋, predict:名古屋
ratio:1.0, answer:徳島, predict:徳島
ratio:1.0, answer:つくば, predict:つくば
ratio:1.0, answer:宮崎, predict:宮崎
ratio:1.0, answer:宇都宮, predict:宇都宮
ratio:1.0, answer:久留米, predict:久留米
ratio:1.0, answer:尾張小牧, predict:尾張小牧
ratio:1.0, answer:保劣跡幡, predict:保劣跡幡
ratio:1.0, answer:湘南, predict:湘南
ratio:1.0, answer:佐世保, predict:佐世保
ratio:0.6666666666666667, answer:術匹題, predict:術四題
ratio:1.0, answer:和泉, predict:和泉
ratio:1.0, answer:和歌山, predict:和歌山
ratio:1.0, answer:姫路, predict:姫路
ratio:1.0, answer:凄魂, predict:凄魂
ratio:1.0, answer:北九州, predict:北九州
Average Ratio: 0.9848484848484849
```