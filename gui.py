from abc import ABC, abstractmethod

from PyQt5.QtWidgets import (QWidget, QGroupBox, 
                             QVBoxLayout, QHBoxLayout, 
                             QPushButton)

from augmented_images import ImageAugmentationer
from train import Trainer
from inference import Inferencer


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
        self.trainer = Trainer()
        self.inferencer = Inferencer()
        
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
        inference_button.clicked.connect(lambda: self._show_inference_plot())
        layout.addWidget(inference_button)

        group.setLayout(layout)
        
        group.setStyleSheet("QGroupBox { border: none; }")

        return group
    
    def _handle_show_training_images(self) -> None:
        self.image_augmentationer.show_original_augmented_compare_plot()
    
    def _handle_show_model_structure(self) -> None:
        self.trainer.show_netG_structure()
        self.trainer.show_netD_structure()
        
    def _handle_show_training_loss(self) -> None:
        self.trainer.show_loss_plot()
    
    def _show_inference_plot(self) -> None:
        self.inferencer.show_inference_plot()