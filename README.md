## Captcha Solver

### Necessário:
tensorflow
object detection (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

### Executando:
1) python3 xml_to_csv.py  ---- Converte os .xlm gerados a partir da rotulação para um csv test e um csv train
2) python3 generate_tfrecord.py --csv_input=data/train_labels.csv --image_dir=train --output_path=data/train.record ---- Converte o csv para o formato .record, que será usado no treinamento
3) python3 generate_tfrecord.py --csv_input=ABC/test_labels.csv --image_dir=test --output_path=data/test.record
4) python3 train.py --logtostderr --train_dir=train_result/ --pipeline_config_path=treinando/faster_rcnn_inception_v2_coco.config
5) python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path data/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix train_result/model.ckpt-XXXX --output_directory inference_graph
6) python3 test.py imagem.png



