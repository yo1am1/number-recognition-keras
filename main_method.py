import dataclasses
import os
from typing import Optional

import cv2
import keras
import numpy as np

from crop import crop_images


@dataclasses.dataclass(frozen=True)
class RecognitionModel:
    model_path: Optional[str]

    input_shape: Optional[tuple] = (28, 28, 1)
    num_classes: Optional[int] = 10
    loss: Optional[str] = "categorical_crossentropy"
    metrics: Optional[list] = "accuracy"
    epochs: Optional[int] = 50
    batch_size: Optional[int] = 2

    logs: Optional[bool] = False
    tensor: Optional[bool] = False
    to_file: Optional[bool] = False

    new_model_name: Optional[str] = "new_model"
    checkpoint: Optional[bool] = True
    checkpoint_path: Optional[str] = "best_model_first"
    early_stopping: Optional[bool] = True
    tensorboard: Optional[bool] = True
    lr_scheduler: Optional[bool] = True
    validation_split: Optional[float] = 0.2

    def __str__(self):
        return f"List of pretrained models: {['multiclass_classification_model.keras']}, RecognitionModel: {self.model_path}, Input shape: {self.input_shape}, Number of classes: {self.num_classes}, Loss: {self.loss}, Metrics: {self.metrics}, Epochs: {self.epochs}, Batch size: {self.batch_size}"


class Recognizer:
    def __init__(self, model: RecognitionModel, crop: bool = False):
        self.__cropped = None
        self.model = model
        self.crop_flag = crop

    def train(self) -> None:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        y_train = keras.utils.to_categorical(y_train, self.model.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.model.num_classes)

        model = keras.Sequential(
            [
                keras.Input(
                    shape=self.model.input_shape, batch_size=self.model.batch_size
                ),
                keras.layers.Flatten(),
                keras.layers.Dense(28 * 28, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(self.model.num_classes, activation="softmax"),
            ]
        )

        model.summary()

        checkpoint_cb = (
            keras.callbacks.ModelCheckpoint(
                f"{self.model.checkpoint_path}.keras",
                save_best_only=True,
                monitor="accuracy",
            )
            if self.model.checkpoint
            else None
        )
        early_stopping_cb = (
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=30, restore_best_weights=True
            )
            if self.model.early_stopping
            else None
        )
        tensorboard_cb = (
            keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)
            if self.model.tensor
            else None
        )
        lr_scheduler_cb = (
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            if self.model.lr_scheduler
            else None
        )

        optimizer = keras.optimizers.Adam()
        model.compile(
            optimizer=optimizer, loss=f"{self.model.loss}", metrics=["accuracy"]
        )

        try:
            model.fit(
                x_train,
                y_train,
                epochs=self.model.epochs,
                batch_size=self.model.batch_size,
                validation_split=self.model.validation_split,
                callbacks=[
                    checkpoint_cb,
                    early_stopping_cb,
                    tensorboard_cb,
                    lr_scheduler_cb,
                ],
            )

            score = model.evaluate(x_test, y_test, verbose=1)
            if self.model.logs:
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

            model.save(f"{self.model.new_model_name}.keras")
        except Exception as e:
            print(f"Error training model: {e}")

    @staticmethod
    def __load_model_val(model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Path {model_path} does not exist.")
        try:
            keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")

    @staticmethod
    def __image_optimizer(image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
        image_resized = cv2.bitwise_not(image_resized)
        test_image = np.array(image_resized)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image.astype("float32") / 255
        return test_image

    def crop(self, path: str) -> None:
        if self.crop_flag:
            try:
                crop_images(path)
                print(f"Images cropped: {path}") if self.model.logs else None
            except Exception as e:
                print(f"Error cropping images: {path}")
                print(e)
                pass

            self.__cropped = True
        else:
            print(f"Wrong crop flag: {self.crop_flag}")
            self.__cropped = False

    def recognize(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")

        if self.crop_flag:
            self.crop(path)

        keras.models.load_model(
            self.model.model_path
        ).summary() if self.model.logs else None

        for file in os.listdir(path):
            file_path = os.path.join(path, file)

            try:
                input_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

                if input_image is None:
                    print(f"Error reading image: {file_path}")
                    continue

                test_image = self.__image_optimizer(input_image)

                self.__load_model_val(self.model.model_path)

                prediction = keras.models.load_model(self.model.model_path).predict(
                    test_image
                )
                print(
                    f"File: {file_path}, Prediction: {np.argmax(prediction)}, Reality: {file[-5]}, Probability: {np.max(prediction)}"
                ) if self.model.logs else None
                if self.model.to_file:
                    with open("results.txt", "a") as f:
                        f.write(
                            f"File: {file_path}, Prediction: {np.argmax(prediction)}, Reality: {file[-5]}, Probability: {np.max(prediction)}\n"
                        )
            except Exception as e:
                print(f"Error processing image: {file_path}")
                print(e)
        f.close()
