from main_method import RecognitionModel, Recognizer

model = RecognitionModel(
    model_path="multiclass_classification_model.keras",
    logs=True,
    tensor=True,
    to_file=True,
)

recognizer = Recognizer(model=model, crop=True)
print(recognizer)

recognizer.recognize("./test_images/")
