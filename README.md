# Turtles Inference

Playground server

## Run

```
pip install -r requirements.txt
python main.py
```

Run a prediction:

```
curl -X POST -F image="@~/path_to_image/FP22-270 R.JPG" "http://localhost:5000/predict"
```