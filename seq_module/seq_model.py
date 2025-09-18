import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from manage.mertic import caculate_metric
from manage.LKA import LKABlock


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, Bilstm_input_feature_size)
        self.pos_embed = nn.Embedding(max_len, Bilstm_input_feature_size)
        self.norm = nn.LayerNorm(Bilstm_input_feature_size)
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(1e-5)
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = torch.nn.Linear(Bilstm_output_feature_size*2, d_k * n_head)
        self.W_K = torch.nn.Linear(Bilstm_output_feature_size*2, d_k * n_head)
        self.W_V = torch.nn.Linear(Bilstm_output_feature_size*2, d_v * n_head)
        self.linear = torch.nn.Linear(n_head * d_v, Bilstm_output_feature_size*2)
        self.norm = torch.nn.LayerNorm(Bilstm_output_feature_size*2)
    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)
        context ,attn= ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,n_head * d_v)
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attn

class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = torch.nn.Linear(Bilstm_output_feature_size*2, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, Bilstm_output_feature_size*2)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None
    def forward(self, enc_inputs):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs,self.attention_map

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate=None):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(growth_rate)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate) 
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out
class DenseBlock(nn.Module):
    def __init__(self, layers, in_channels, growth_rate, dropout_rate=None):
        super(DenseBlock, self).__init__()
        self.layers = layers
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate 
        self.dense_layers = nn.ModuleList()
        for i in range(layers):
            self.dense_layers.append(DenseLayer(in_channels, growth_rate, dropout_rate))
            in_channels += growth_rate  # Update input channels for the next layer
    def forward(self, x):
        feature_maps = [x]
        for layer in self.dense_layers:
            out = layer(x)
            feature_maps.append(out)
            x = torch.cat(feature_maps, dim=1)  # Concatenate feature maps along the channel dimension
        return x
    
class PLMEncoder(nn.Module):
    def __init__(self, BERT_encoder, out_dim, PLM_dim=1024, dropout=0.5):
        super(PLMEncoder, self).__init__()
        self.bert = BERT_encoder # BertModel.from_pretrained("Rostlab/prot_bert")
        for param in self.bert.base_model.parameters():
            param.requires_grad = False
        self.lkablock=LKABlock(PLM_dim)
        self.learn = torch.nn.Sequential(
            torch.nn.Linear( PLM_dim, PLM_dim//4),
            torch.nn.Dropout(0.4),
            torch.nn.ReLU(),
            torch.nn.Linear( PLM_dim//4, PLM_dim//16),
            torch.nn.Dropout(0.4),
            torch.nn.ReLU(),
            torch.nn.Linear( PLM_dim//16, PLM_dim//32))

    def forward(self, input_ids, attention_mask):
        pooled_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output=self.lkablock(pooled_output)
        output = pooled_output.permute(0, 2, 1) # shape: (Batch, 1024, length)
        prot_out = torch.mean(output, axis=-1) # shape: (Batch, out_channel, 1)
        prot_out=self.learn(prot_out)
        return prot_out
    
class seq_encoder(torch.nn.Module):
    def __init__(self,Bilstm_input_feature_size_1,Bilstm_output_feature_size_1,Bilstm_layer_number_1,n_layers_1,n_head_1,d_k_1,CNN_out_channels_1,conv1_kernel_size_1,conv1_padding_1   ,max_len_1  , vocab_size_1  ,device_1 ):
        super(seq_encoder,self).__init__()
        global Bilstm_input_feature_size, Bilstm_output_feature_size,Bilstm_layer_number, n_layers,  n_head, d_k, d_v, d_ff,CNN_out_channels, conv1_kernel_size, conv1_padding,linear1_input_size,class_number ,max_len  , vocab_size ,device
        Bilstm_input_feature_size=Bilstm_input_feature_size_1
        Bilstm_output_feature_size=Bilstm_output_feature_size_1
        Bilstm_layer_number=Bilstm_layer_number_1
        n_layers = n_layers_1
        n_head = n_head_1
        d_k = d_k_1
        d_v = d_k_1
        d_ff = d_k_1
        CNN_out_channels=CNN_out_channels_1
        conv1_kernel_size=conv1_kernel_size_1
        conv1_padding=conv1_padding_1
        linear1_input_size = CNN_out_channels * 6
        class_number = 2
        max_len, vocab_size, device= max_len_1, vocab_size_1, device_1
        self.embedding = Embedding()
        self.Bilstm = torch.nn.LSTM(Bilstm_output_feature_size, Bilstm_output_feature_size, Bilstm_layer_number, batch_first=True, bidirectional=True)
        self.layers =  torch.nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.lkablock=LKABlock(Bilstm_output_feature_size*2)
        self.block=DenseBlock(n_layers,Bilstm_output_feature_size*2,CNN_out_channels,0.5)
        self.pool = torch.nn.AvgPool1d(2, stride=2, )
        self.dropout = torch.nn.Dropout(0.4)
        self.feature_learn = torch.nn.Sequential(
            torch.nn.Linear( Bilstm_output_feature_size*2+n_layers*CNN_out_channels, linear1_input_size//4),
            torch.nn.Dropout(0.4),
            torch.nn.ReLU(),)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(class_number, 1))

    def forward(self, input_ids,x6):
        attention_weights = []
        input_ids = input_ids.to(device)
        outputembedding = self.embedding(input_ids.to(device))
        outputembedding=torch.cat([x6, outputembedding], axis=-1)
        output , (h_n, c_n) =self.Bilstm(outputembedding)
        for layer in self.layers:
            output_after_attention,att = layer(output)
            attention_weights.append(att)
            pass
        x1 = output
        x1=self.dropout(x1)
        x2=self.lkablock(x1)
        x=x1+x2
        x=x.permute(0,2,1)
        x=self.dropout(x)
        x=self.block(x)
        x=self.dropout(x)
        x = self.pool(x)
        x = torch.mean(x, dim=-1)
        feature_learn_seq_encoder =self.feature_learn(x)
        return feature_learn_seq_encoder,attention_weights

class combine_model(nn.Module):
    def __init__(self,BERT_encoder, Bilstm_input_feature_size_1, Bilstm_output_feature_size_1, Bilstm_layer_number_1, n_layers_1,
                 n_head_1, d_k_1, CNN_out_channels_1, conv1_kernel_size_1, conv1_padding_1     ,MLP_input_dim  , max_len  , vocab_size  ,device  ):
        super(combine_model,self).__init__()
        self.seq_encoder=seq_encoder(Bilstm_input_feature_size_1, Bilstm_output_feature_size_1, Bilstm_layer_number_1, n_layers_1,
                 n_head_1, d_k_1, CNN_out_channels_1, conv1_kernel_size_1, conv1_padding_1      , max_len  , vocab_size  ,device      )
        self.plm_encoder = PLMEncoder(BERT_encoder=BERT_encoder, out_dim=32, PLM_dim=1024, dropout=0.4)
        self.feature_learn_combine = torch.nn.Sequential(
            torch.nn.Linear(((linear1_input_size // 4)  +28+32),
                            ((linear1_input_size // 4)  +28 +32) // 2),
            nn.BatchNorm1d(((linear1_input_size // 4)  +28 +32) // 2),  # 添加批量归一化
            torch.nn.ReLU(),
            torch.nn.Linear(((linear1_input_size // 4)+28 +32) // 2, class_number) )
        self.norm = nn.LayerNorm((linear1_input_size // 4)  +28+32)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(class_number, 1),
            nn.Sigmoid())
    def forward(self, x1,x2,x3,x4,x5):
        x_2,atts = self.seq_encoder(x2,x1)
        x5=self.plm_encoder(x4,x5)
        feature_combine_two_model = torch.cat([x_2,x3,x5], axis=1)
        feature_combine_two_model=self.norm(feature_combine_two_model)
        feature_learn_combine = self.feature_learn_combine(feature_combine_two_model)
        feature_combine_two_model=nn.Sigmoid()(feature_combine_two_model)
        logit= self.classifier (feature_learn_combine)
        return logit,feature_learn_combine,feature_combine_two_model,atts


def test(model: torch.nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob_positive = torch.empty([0], device=device)
    repres_list=[]
    label_list=[]
    all_att=[]
    combine_features=torch.empty([0], device=device)
    all_emb=[]
    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            x[0] = (x[0].float()).to(device)
            x[2] = (x[2].float()).to(device)
            x[3] = x[3].to(device)
            x[4] = x[4].to(device)
            y = (y.float()).to(device)
            class_fenlei,representation,combine_feature,atts = model(*x)
            test_loss += loss_function(class_fenlei.squeeze(),y)
            class_fenlei=torch.cat(( 1-class_fenlei,class_fenlei), 1)
            pred_prob_positive_batch = class_fenlei[:, 1]  #
            pred_class= np.zeros(( class_fenlei.size(0),1))
            for i in range(len(pred_prob_positive_batch)):
                pred_class[i,0] = int(pred_prob_positive_batch[i]>0.5)
            pred_class=torch.from_numpy(pred_class)
            pred_class=torch.squeeze(pred_class, dim=1).to(device)
            true_label =y
            label_pred = torch.cat([label_pred, pred_class.float()])
            label_real = torch.cat([label_real, true_label.float()])
            pred_prob_positive = torch.cat([pred_prob_positive, pred_prob_positive_batch])
            combine_features=torch.cat([combine_features, combine_feature])
            atts = torch.stack(atts)
            all_att.append(atts)
            all_emb.extend(x[1].detach().cpu().numpy())
            repres_list.extend(representation.cpu().detach().numpy())
            label_list.extend(y.cpu().detach().numpy())
        metric,roc_data,aupr_data = caculate_metric(label_pred, label_real, pred_prob_positive)
        combine_features=combine_features.cpu().detach().numpy()
        pro_1=pred_prob_positive.cpu().detach().numpy()
        true_sample_label=label_real.cpu().detach().numpy()
        test_loss /= len(test_loader.dataset)
        evaluation = {'loss': test_loss, 'else': metric  }
    return evaluation,repres_list,label_list,roc_data,aupr_data,true_sample_label, pro_1,combine_features,all_emb