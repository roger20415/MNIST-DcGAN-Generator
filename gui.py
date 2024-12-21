from abc import ABC, abstractmethod

from PyQt5.QtWidgets import (QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSizePolicy, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib import image

from augmented_images import ImageAugmentationer
from config import Config


class Gui(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("MNIST-DcGAN-Generator")
        self.setGeometry(100, 100, 50, 100)

        main_layout = QHBoxLayout()

        button_column = ButtonColumn(self)

        main_layout.addWidget(button_column.create_column())
        self.setLayout(main_layout)
        return None


class BaseColumn(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_column(self) -> QGroupBox:
        pass


class ButtonColumn(BaseColumn):
    def __init__(self, parent_widget) -> None:
        self._parent_widget = parent_widget
        self._inference_image_path: str = None
        self.image_augmentationer = ImageAugmentationer()
        
    def create_column(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout()

        show_training_images_button = QPushButton("Show Training Images")
        show_training_images_button.clicked.connect(lambda: self._handle_show_training_images())
        layout.addWidget(show_training_images_button)

        show_model_structure_button = QPushButton("Show Model Structure")
        show_model_structure_button.clicked.connect(lambda: self._handle_show_model_structure())
        layout.addWidget(show_model_structure_button)

        show_training_loss_button = QPushButton("Show Training Loss")
        show_training_loss_button.clicked.connect(lambda: self._handle_show_training_loss())
        layout.addWidget(show_training_loss_button)

        inference_button = QPushButton("Inference")
        inference_button.clicked.connect(lambda: self._handle_inference())
        layout.addWidget(inference_button)

        group.setLayout(layout)
        
        group.setStyleSheet("QGroupBox { border: none; }")

        return group
    
    def _handle_show_training_images(self) -> None:
        self.image_augmentationer.show_original_augmented_compare_plot()
    
    def _handle_show_model_structure(self) -> None:
        pass
        """
        self.vgg_trainer.show_vgg_structure()
        """
    
    def _handle_show_accuracy_loss(self) -> None:
        pass
        """
        acc_img = image.imread(Config.ACC_PLOT_PATH)
        val_img = image.imread(Config.LOSS_PLOT_PATH)

        _, axes = plt.subplots(2, 1, figsize=(6, 9))
        axes[0].imshow(acc_img)
        axes[0].axis('off')
        axes[0].set_title('Accuracy')

        axes[1].imshow(val_img)
        axes[1].axis('off')
        axes[1].set_title('Loss')

        plt.show()
        """
    
    def _show_inference_image(self, image_path: str) -> None:
        pass
        """
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(128, 128, Qt.KeepAspectRatio)
        self.display_column.image_label.setPixmap(pixmap)
        self.display_column.image_label.setAlignment(Qt.AlignLeft)
        """
    
    def _handle_inference(self) -> None:
        pass
        """
        predicted_label: str = self.vgg_inferencer.inference(self._inference_image_path)
        self._show_inference_result(predicted_label)
        self._show_probalility_bar()
        """
    
    def _show_inference_result(self, predicted_label: str) -> None:
        pass
        """
        self.display_column.predict_label.setText(f"Predicted: {predicted_label}")
        """
    
    def _show_probalility_bar(self) -> None:
        pass
        """
        prob_bar = image.imread(Config.PROBABILITY_BAR_PATH)
        plt.imshow(prob_bar)
        plt.axis('off')
        plt.show()
        """