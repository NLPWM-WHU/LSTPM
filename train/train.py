import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import numpy as np
import torch
from collections import defaultdict
import gc
import os
from math import radians, cos, sin, asin, sqrt
from collections import deque,Counter
def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)
    if require_indices:
        return result, shuffle_indices
    else:
        return result

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 128)
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def pad_batch_of_lists_masks(batch_of_lists, max_len):
    padded = [l + [0] * (max_len - len(l)) for l in batch_of_lists]
    padded_mask = [[1.0]*(len(l) - 1) + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padde_mask_non_local = [[1.0] * (len(l)) + [0.0] * (max_len - len(l)) for l in batch_of_lists]
    return padded, padded_mask, padde_mask_non_local

def pad_batch_of_lists_masks_test(batch_of_lists, max_len):
    padded = [l + [0] * (max_len - len(l)) for l in batch_of_lists]
    padded2 = [l[:-1] + [0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padded_mask = [[0.0]*(len(l) - 2) + [1.0] + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padde_mask_non_local = [[1.0] * (len(l) - 1) + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    return padded, padded2, padded_mask, padde_mask_non_local

class Model(nn.Module):
    def __init__(self, n_users, n_items, emb_size=500, hidden_units=500, dropout=0.8, user_dropout=0.5, data_neural = None, tim_sim_matrix = None):
        super(self.__class__, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_units = hidden_units
        if emb_size == None:
            emb_size = hidden_units
        self.emb_size = emb_size
        ## todo why embeding?
        self.item_emb = nn.Embedding(n_items, emb_size)
        self.emb_tim = nn.Embedding(48, 10)
        self.lstmcell = nn.LSTM(input_size=emb_size, hidden_size=hidden_units)
        self.lstmcell_history = nn.LSTM(input_size=emb_size, hidden_size=hidden_units)
        self.linear = nn.Linear(hidden_units*2 , n_items)
        self.dropout = nn.Dropout(0.0)
        self.user_dropout = nn.Dropout(user_dropout)
        self.data_neural = data_neural
        self.tim_sim_matrix = tim_sim_matrix
        self.dilated_rnn = nn.LSTMCell(input_size=emb_size, hidden_size=hidden_units)
        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, user_vectors, item_vectors, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, is_train, poi_distance_matrix, sequence_dilated_rnn_index_batch):
        batch_size = item_vectors.size()[0]
        sequence_size = item_vectors.size()[1]
        items = self.item_emb(item_vectors)
        item_vectors = item_vectors.cpu()
        x = self.dropout(items)
        x = x.transpose(0, 1)
        h1 = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        c1 = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        out, (h1, c1) = self.lstmcell(x, (h1, c1))
        out = out.transpose(0, 1)#batch_size * sequence_length * embedding_dim
        x1 = self.dropout(items)
        # ###########################################################
        user_batch = np.array(user_vectors.cpu())
        y_list = []
        out_hie = []
        for sk in range(batch_size):
            ##########################################
            current_session_input_dilated_rnn_index = sequence_dilated_rnn_index_batch[sk]
            hiddens_current = x1[sk]
            dilated_lstm_outs_h = []
            dilated_lstm_outs_c = []
            for index_dilated in range(len(current_session_input_dilated_rnn_index)):
                index_dilated_explicit = current_session_input_dilated_rnn_index[index_dilated]
                hidden_current = hiddens_current[index_dilated].unsqueeze(0)
                if index_dilated == 0:
                    h = Variable(torch.zeros(1, self.hidden_units)).cuda()
                    c = Variable(torch.zeros(1, self.hidden_units)).cuda()
                    (h, c) = self.dilated_rnn(hidden_current, (h, c))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
                else:
                    (h, c) = self.dilated_rnn(hidden_current, (dilated_lstm_outs_h[index_dilated_explicit], dilated_lstm_outs_c[index_dilated_explicit]))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
            dilated_lstm_outs_h.append(hiddens_current[len(current_session_input_dilated_rnn_index):])
            dilated_out = torch.cat(dilated_lstm_outs_h, dim = 0).unsqueeze(0)
            out_hie.append(dilated_out)
            user_id_current = user_batch[sk]
            current_session_timid = sequence_tim_batch[sk][:-1]
            current_session_poiid = item_vectors[sk][:len(current_session_timid)]
            session_id_current = session_id_batch[sk]
            current_session_embed = out[sk]
            current_session_mask = mask_batch_ix_non_local[sk].unsqueeze(1)
            sequence_length = int(sum(np.array(current_session_mask.cpu()))[0])
            current_session_represent_list = []
            if is_train:
                for skk in range(sequence_length-1):
                    current_session_represent = torch.sum(current_session_embed * current_session_mask, dim=0).unsqueeze(0)/sum(current_session_mask)
                    current_session_represent_list.append(current_session_represent)
            else:
                for skk in range(sequence_length-1):
                    current_session_represent_rep_item = current_session_embed[0:skk+1]
                    current_session_represent_rep_item = torch.sum(current_session_represent_rep_item, dim = 0).unsqueeze(0)/(skk + 1)
                    current_session_represent_list.append(current_session_represent_rep_item)

            current_session_represent = torch.cat(current_session_represent_list, dim = 0)
            list_for_sessions = []
            list_for_avg_distance = []
            h2 = Variable(torch.zeros(1, 1, self.hidden_units)).cuda()###whole sequence
            c2 = Variable(torch.zeros(1, 1, self.hidden_units)).cuda()
            for sk1 in range(session_id_current):
                sequence = [s[0] for s in self.data_neural[user_id_current]['sessions'][sk1]]
                sequence = Variable(torch.LongTensor(np.array(sequence))).cuda()
                sequence_emb = self.item_emb(sequence).unsqueeze(1)
                sequence = sequence.cpu()
                sequence_emb, (h2, c2) = self.lstmcell_history(sequence_emb, (h2, c2))
                sequence_tim_id = [s[1] for s in self.data_neural[user_id_current]['sessions'][sk1]]
                jaccard_sim_row = Variable(torch.FloatTensor(self.tim_sim_matrix[current_session_timid]),requires_grad=False).cuda()
                jaccard_sim_expicit = jaccard_sim_row[:,sequence_tim_id]
                distance_row = poi_distance_matrix[current_session_poiid]
                distance_row_expicit = Variable(torch.FloatTensor(distance_row[:,sequence]),requires_grad=False).cuda()
                distance_row_expicit_avg = torch.mean(distance_row_expicit, dim = 1)
                jaccard_sim_expicit_last = F.softmax(jaccard_sim_expicit)
                hidden_sequence_for_current1 = torch.mm(jaccard_sim_expicit_last, sequence_emb.squeeze(1))
                hidden_sequence_for_current =  hidden_sequence_for_current1
                list_for_sessions.append(hidden_sequence_for_current.unsqueeze(0))
                list_for_avg_distance.append(distance_row_expicit_avg.unsqueeze(0))
            avg_distance = torch.cat(list_for_avg_distance, dim = 0).transpose(0,1)
            sessions_represent = torch.cat(list_for_sessions, dim=0).transpose(0,1) ##current_items * history_session_length * embedding_size
            current_session_represent = current_session_represent.unsqueeze(2) ### current_items * embedding_size * 1
            sims = F.softmax(sessions_represent.bmm(current_session_represent).squeeze(2), dim = 1).unsqueeze(1) ##==> current_items * 1 * history_session_length
            out_y_current = sims.bmm(sessions_represent).squeeze(1)
            ##############layer_2
            layer_2_current = (out_y_current + current_session_embed[:sequence_length-1]).unsqueeze(2)##==>current_items * embedding_size * 1
            layer_2_sims =  F.softmax(sessions_represent.bmm(layer_2_current).squeeze(2) * 1.0/avg_distance, dim = 1).unsqueeze(1)##==>>current_items * 1 * history_session_length
            out_layer_2 = layer_2_sims.bmm(sessions_represent).squeeze(1)
            out_y_current_padd = Variable(torch.FloatTensor(sequence_size - sequence_length + 1, self.emb_size).zero_(),requires_grad=False).cuda()
            out_layer_2_list = []
            out_layer_2_list.append(out_layer_2)
            out_layer_2_list.append(out_y_current_padd)
            out_layer_2 = torch.cat(out_layer_2_list,dim = 0).unsqueeze(0)
            y_list.append(out_layer_2)
        y = torch.cat(y_list,dim=0)
        out_hie = F.selu(torch.cat(out_hie, dim = 0))
        out = F.selu(out)
        out = (out + out_hie) * 0.5
        out_put_emb_v1 = torch.cat([y, out], dim=2)
        out_put_emb_v1 = self.dropout(out_put_emb_v1)
        output_ln = self.linear(out_put_emb_v1)
        output = F.log_softmax(output_ln, dim=-1)
        return output




def caculate_time_sim(data_neural):
    time_checkin_set = defaultdict(set)
    for uid in data_neural:
        uid_sessions = data_neural[uid]
        for sid in uid_sessions['sessions']:
            session_current = uid_sessions['sessions'][sid]
            for checkin in session_current:
                timid = checkin[1]
                locid = checkin[0]
                if timid not in time_checkin_set:
                    time_checkin_set[timid] = set()
                time_checkin_set[timid].add(locid)
    sim_matrix = np.zeros((48,48))
    for i in range(48):
        for j in range(48):
            set_i = time_checkin_set[i]
            set_j = time_checkin_set[j]
            jaccard_ij = len(set_i & set_j)/len(set_i | set_j)
            sim_matrix[i][j] = jaccard_ij
    return sim_matrix

def caculate_poi_distance(poi_coors):
    sim_matrix = np.zeros((len(poi_coors) + 1, len(poi_coors) + 1))
    for i in range(len(poi_coors)):
        print(i)
        for j in range(i , len(poi_coors)):
            poi_current = i + 1
            poi_target = j + 1
            poi_current_coor = poi_coors[poi_current]
            poi_target_coor = poi_coors[poi_target]
            distance_between = geodistance(poi_current_coor[1], poi_current_coor[0], poi_target_coor[1], poi_target_coor[0])
            if distance_between<1:
                distance_between = 1
            sim_matrix[poi_current][poi_target] = distance_between
            sim_matrix[poi_target][poi_current] = distance_between
    pickle.dump(sim_matrix, open('distance.pk', 'wb'))
    return sim_matrix

def generate_input_history(data_neural, mode, mode2=None, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            target = np.array([s[0] for s in session[1:]])
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['target'] = Variable(torch.LongTensor(target))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)
            if mode2 == 'max':
                history_tmp = {}
                for tr in history:
                    if tr[1] not in history_tmp:
                        history_tmp[tr[1]] = [tr[0]]
                    else:
                        history_tmp[tr[1]].append(tr[0])
                history_filter = []
                for t in history_tmp:
                    if len(history_tmp[t]) == 1:
                        history_filter.append((history_tmp[t][0], t))
                    else:
                        tmp = Counter(history_tmp[t]).most_common()
                        if tmp[0][1] > 1:
                            history_filter.append((history_tmp[t][0], t))
                        else:
                            ti = np.random.randint(len(tmp))
                            history_filter.append((tmp[ti][0], t))
                history = history_filter
                history = sorted(history, key=lambda x: x[1], reverse=False)
            elif mode2 == 'avg':
                history_tim = [t[1] for t in history]
                history_count = [1]
                last_t = history_tim[0]
                count = 1
                for t in history_tim[1:]:
                    if t == last_t:
                        count += 1
                    else:
                        history_count[-1] = count
                        history_count.append(1)
                        last_t = t
                        count = 1
            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            if mode2 == 'avg':
                trace['history_count'] = history_count
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])
            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history_tim = [t[1] for t in history]
            history_count = [1]
            last_t = history_tim[0]
            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1
            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_count'] = history_count
            loc_tim = history
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target'] = Variable(torch.LongTensor(target))
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

def generate_queue(train_idx, mode, mode2):
    user = list(train_idx.keys())
    train_queue = list()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def create_dilated_rnn_input(session_sequence_current, poi_distance_matrix):
    sequence_length = len(session_sequence_current)
    session_sequence_current.reverse()
    session_dilated_rnn_input_index = [0] * sequence_length
    for i in range(sequence_length - 1):
        current_poi = [session_sequence_current[i]]
        poi_before = session_sequence_current[i + 1 :]
        distance_row = poi_distance_matrix[current_poi]
        distance_row_explicit = distance_row[:, poi_before][0]
        index_closet = np.argmin(distance_row_explicit)
        session_dilated_rnn_input_index[sequence_length - i - 1] = sequence_length-2-index_closet-i
    return session_dilated_rnn_input_index



def generate_detailed_batch_data(one_train_batch):
    session_id_batch = []
    user_id_batch = []
    sequence_batch = []
    sequences_lens_batch = []
    sequences_tim_batch = []
    sequences_dilated_input_batch = []
    for sample in one_train_batch:
        user_id_batch.append(sample[0])
        session_id_batch.append(sample[1])
        session_sequence_current = [s[0] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_sequence_tim_current = [s[1] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_sequence_dilated_input = create_dilated_rnn_input(session_sequence_current, poi_distance_matrix)
        sequence_batch.append(session_sequence_current)
        sequences_lens_batch.append(len(session_sequence_current))
        sequences_tim_batch.append(session_sequence_tim_current)
        sequences_dilated_input_batch.append(session_sequence_dilated_input)
    return user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequences_tim_batch, sequences_dilated_input_batch


def train_network(network, num_epoch=40 ,batch_size = 32,criterion = None):
    candidate = data_neural.keys()
    data_train, train_idx = generate_input_history(data_neural, 'train', candidate=candidate)
    for epoch in range(num_epoch):
        network.train(True)
        i = 0
        run_queue = generate_queue(train_idx, 'random', 'train')
        for one_train_batch in minibatch(run_queue, batch_size = batch_size):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch, sequence_dilated_rnn_index_batch = generate_detailed_batch_data(one_train_batch)
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, mask_batch_ix, mask_batch_ix_non_local = pad_batch_of_lists_masks(sequence_batch,
                                                                                                     max_len)
            padded_sequence_batch = Variable(torch.LongTensor(np.array(padded_sequence_batch))).to(device)
            mask_batch_ix = Variable(torch.FloatTensor(np.array(mask_batch_ix))).to(device)
            mask_batch_ix_non_local = Variable(torch.FloatTensor(np.array(mask_batch_ix_non_local))).to(device)
            user_id_batch = Variable(torch.LongTensor(np.array(user_id_batch))).to(device)
            logp_seq = network(user_id_batch, padded_sequence_batch, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, True, poi_distance_matrix, sequence_dilated_rnn_index_batch)
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
            loss = -logp_next.sum() / mask_batch_ix[:, :-1].sum()
            # train with backprop
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            opt.step()
            if (i + 1) % 20 == 0:
                print("epoch" + str(epoch) + ": loss: " + str(loss))
            i += 1
        acc10 = evaluate(network, 10, 1)
        print("ACC@ score: ", acc10)


def get_acc(target, scores):
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    ndcg = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t != 0:
            if t in p[:10] and t > 0:
                acc[0] += 1
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                acc[1] += 1
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                acc[2] += 1
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)
        else:
            break
    return acc.tolist(), ndcg.tolist()

def evaluate(network, k, batch_size = 2):
    network.train(False)
    candidate = data_neural.keys()
    data_test, test_idx = generate_input_long_history(data_neural, 'test', candidate=candidate)
    users_acc = {}
    with torch.no_grad():
        run_queue = generate_queue(test_idx, 'normal', 'test')
        for one_test_batch in minibatch(run_queue, batch_size=batch_size):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch, sequence_dilated_rnn_index_batch = generate_detailed_batch_data(
                one_test_batch)
            user_id_batch_test = user_id_batch
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, mask_batch_ix, mask_batch_ix_non_local = pad_batch_of_lists_masks(sequence_batch,
                                                                                                     max_len)
            padded_sequence_batch = Variable(torch.LongTensor(np.array(padded_sequence_batch))).to(device)
            mask_batch_ix = Variable(torch.FloatTensor(np.array(mask_batch_ix))).to(device)
            mask_batch_ix_non_local = Variable(torch.FloatTensor(np.array(mask_batch_ix_non_local))).to(device)
            user_id_batch = Variable(torch.LongTensor(np.array(user_id_batch))).to(device)
            logp_seq = network(user_id_batch, padded_sequence_batch, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, False, poi_distance_matrix, sequence_dilated_rnn_index_batch)
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            for ii, u_current in enumerate(user_id_batch_test):
                if u_current not in users_acc:
                    users_acc[u_current] = [0, 0, 0, 0, 0, 0, 0]
                acc, ndcg = get_acc(actual_next_tokens[ii], predictions_logp[ii])
                users_acc[u_current][1] += acc[2][0]#@1
                users_acc[u_current][2] += acc[1][0]#@5
                users_acc[u_current][3] += acc[0][0]#@10
                ###ndcg
                users_acc[u_current][4] += ndcg[2][0]  # @1
                users_acc[u_current][5] += ndcg[1][0]  # @5
                users_acc[u_current][6] += ndcg[0][0]  # @10
                users_acc[u_current][0] += (sequences_lens_batch[ii]-1)
        tmp_acc = [0.0,0.0,0.0, 0.0, 0.0, 0.0]##last 3 ndcg
        sum_test_samples = 0.0
        for u in users_acc:
            tmp_acc[0] = users_acc[u][1] + tmp_acc[0]
            tmp_acc[1] = users_acc[u][2] + tmp_acc[1]
            tmp_acc[2] = users_acc[u][3] + tmp_acc[2]

            tmp_acc[3] = users_acc[u][4] + tmp_acc[3]
            tmp_acc[4] = users_acc[u][5] + tmp_acc[4]
            tmp_acc[5] = users_acc[u][6] + tmp_acc[5]
            sum_test_samples = sum_test_samples + users_acc[u][0]
        avg_acc = (np.array(tmp_acc)/sum_test_samples).tolist()
        return avg_acc

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data = pickle.load(open('foursquare_cut_one_day.pkl', 'rb'), encoding='iso-8859-1')
    vid_list = data['vid_list']
    uid_list = data['uid_list']
    data_neural = data['data_neural']
    poi_coordinate = data['vid_lookup']
    loc_size = len(vid_list)
    uid_size = len(uid_list)
    time_sim_matrix = caculate_time_sim(data_neural)
    poi_distance_matrix = pickle.load(open('distance.pkl', 'rb'), encoding='iso-8859-1')
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")
    n_users = uid_size
    n_items = loc_size
    session_id_sequences = None
    user_id_session = None
    network = Model(n_users=n_users, n_items=n_items, data_neural=data_neural, tim_sim_matrix=time_sim_matrix).to(
        device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=0.0001,
                               weight_decay=1 * 1e-6)
    criterion = nn.NLLLoss().cuda()
    train_network(network,criterion=criterion)
