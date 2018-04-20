python /code/simpson_retraining/label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Mul \
--output_layer=final_result \
--image=$1