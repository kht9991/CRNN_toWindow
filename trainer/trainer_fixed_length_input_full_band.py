import matplotlib.pyplot as plt
import numpy as np
import torch

from inferencer.inferencer import inference_wrapper
from trainer.base_trainer import BaseTrainer

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, config, resume: bool, model, loss_function, optimizer, train_dataloader, validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for noisy_mag, clean_mag, _, name in self.train_dataloader: # 언패킹 수 맞춰주기
            noisy_mag = noisy_mag.to(self.device)
            clean_mag = clean_mag.to(self.device)

            self.optimizer.zero_grad()
            
            # 데이터 형태를 확인
            print(f"Original shape of noisy_mag: {noisy_mag.shape}")

            # 데이터 형태를 (batch_size, sequence_length, input_size)로 변환
            # 여기서는 임의로 배치 크기를 설정하고, input_size를 1024로 맞추기 위해 sequence_length를 계산합니다
            batch_size = noisy_mag.size(0)
            sequence_length = noisy_mag.size(1) * noisy_mag.size(2) // 1024
            
            # 데이터 형태 변환
            noisy_mag = noisy_mag.view(batch_size, sequence_length, 1024)

            print(f"Transformed shape of noisy_mag: {noisy_mag.shape}")
            enhanced_mag = self.model(noisy_mag)

            loss = self.loss_function(clean_mag, enhanced_mag)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        self.writer.add_scalar(f"Train/Loss", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        noisy_list, enhanced_list, clean_list, name_list, loss = inference_wrapper(
            dataloader=self.validation_dataloader,
            model=self.model,
            loss_function=self.loss_function,
            device=self.device,
            inference_args=self.validation_custom_config,
            enhanced_dir=None
        )

        self.writer.add_scalar(f"Validation/Loss", loss, epoch)

        for i in range(np.min([self.validation_custom_config["visualization_limit"], len(self.validation_dataloader)])):
            self.spec_audio_visualization(
                noisy_list[i],
                enhanced_list[i],
                clean_list[i],
                name_list[i],
                epoch
            )

        score = self.metrics_visualization(noisy_list, clean_list, enhanced_list, epoch)
        return score
