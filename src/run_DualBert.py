import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random,json,sys
from modeling_DualBert import DualEncoderModel
from model_MocoBert import MoCoBERT
from transformers import (
    HfArgumentParser,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer)
from arguments import ModelArguments, DataTrainingArguments, DELTA_PreTrainingArguments as TrainingArguments
from transformers.trainer import Trainer

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.mse_loss = nn.MSELoss()

    def forward(self, output_A, output_B, labels):
        # 计算余弦相似度
        similarity = self.cosine_similarity(output_A, output_B)
        # 计算损失
        loss = self.mse_loss(similarity, labels.float())
        return loss

class DELTA_Collator(DataCollatorForWholeWordMask):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def convert_to_features(self, dataset):
        src_texts = []
        trgA_texts = []
        trgB_texts = []
        for text_dict in dataset:
            # for i in text_dict:
            if isinstance(text_dict, dict):
                src_texts.append(text_dict['fact'])
                trgA_texts.append(text_dict['interpretation'])
                trgB_texts.append(text_dict['judgment'])
            else:
                print(text_dict)
        input_encodings = self.tokenizer.batch_encode_plus(
            src_texts,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        targetA_encodings = self.tokenizer.batch_encode_plus(
            trgA_texts,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        # targetB_encodings = self.tokenizer.batch_encode_plus(
        #     trgB_texts,
        #     truncation=True,
        #     padding='max_length',
        #     max_length=1024,
        #     return_tensors='pt'
        # )

        encodings = {
            'input_ids_A': input_encodings['input_ids'],
            'attention_mask_A': input_encodings['attention_mask'],
            'input_ids_B': targetA_encodings['input_ids'],
            'attention_mask_B': targetA_encodings['attention_mask'],
            # 'decoder_B_input_ids': targetB_encodings['input_ids'],
            # 'decoder_B_attention_mask': targetB_encodings['attention_mask']
        }
        return encodings

    def __call__(self, examples):
        return self.convert_to_features(examples)

class DELTA_Dataset(Dataset):
    def __init__(self, data_path):
        self.dataset = []
        f = open(data_path, "r", encoding="utf8")
        for line in tqdm(f):
            tem = json.loads(line)
            if tem['interpretation']:
                self.dataset.append({'fact':tem['fact'],'interpretation':tem['interpretation'],'judgment':tem['judgment']})
        self.rng = random.Random()
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

def main():
    # # Training
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = DualEncoderModel()
    model = MoCoBERT().to(device)
    tokenizer = AutoTokenizer.from_pretrained("/jupyterhub_data/test9/model/bert-base-chinese")
    data_collator = DELTA_Collator(tokenizer = tokenizer)
    train_dataloader = DELTA_Dataset(data_args.train_path)
    eval_dataloader = DELTA_Dataset(data_args.eval_path)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train(training_args.resume_from_checkpoint)

    # save model
    #trainer.save_model(data_args.model_save_path)

    torch.save(model.fact_encoder.state_dict(), 'fact_encoder.pth')
    torch.save(model.reason_encoder.state_dict(), 'reason_encoder.pth')
    torch.save(model.state_dict(), 'best_model.pth')
    print("Best model saved!")

if __name__ == "__main__":
    main()
