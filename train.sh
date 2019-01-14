read -p "Enter the pretrained model name: " model_name
python -c "import model_tensorflow; train("$model_name")"
