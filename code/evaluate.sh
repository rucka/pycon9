python /code/simpson_retraining/label_image.py \
--graph=/models/simpson_graph_4000.pb --labels=/models/simpson_labels.txt \
--input_layer=Mul \
--output_layer=final_result \
--image=$1