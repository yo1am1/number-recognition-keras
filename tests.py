import os

import crop
import main_method


def test_crop_images():
    """
    Test for crop_images() function.
    :return None:
    """
    path = "./test_images/"
    try:
        crop.crop_images(path)
    except Exception as e:
        print(f"Error cropping images: {path}")
        print(e)
        pass

    assert os.path.exists(path)


def test_recognize():
    """
    Test for recognize() function.
    :return None:
    """
    path = "./test_images/"

    model = main_method.RecognitionModel(
        model_path="multiclass_classification_model.keras",
        logs=True,
        tensor=True,
        to_file=True,
    )

    recognizer = main_method.Recognizer(model=model, crop=True)
    print(recognizer)

    try:
        recognizer.recognize(path)
    except Exception as e:
        print(f"Error recognizing images: {path}")
        print(e)
        pass

    assert os.path.exists(path)
