    # with open(file_path, 'rb') as f:
    #     features_labels = pickle.load(f)
    # feature_dataset_decode = data_utils.FeatureDataset_decode(args, features_labels, train_loader_list)
    # feature_dataloader_decode = torch.utils.data.DataLoader(feature_dataset_decode, batch_size=args.batch, shuffle=True)  
    # decode_loss_func = nn.MSELoss()
    # Res = tmodels.resnet18(weights=tmodels.resnet18_Weights.DEFAULT)
    # model_f = nn.Sequential(*list(Res.children())[:-1]).to(args.device)
    # model_f.eval()
    # input_image_reconstructed = []  # 随机初始化图像
    # for i in tqdm(range(100)):
    #     loss = []
    #     for idx, (y, _, _ ) in enumerate(feature_dataloader_decode):
    #         y = y.to(args.device).float()
    #         if i == 0:
    #             image_reconstructed = torch.randn(y.size(0), 3, args.size, args.size).to(args.device)
    #             image_reconstructed.requires_grad = True
    #         else:
    #             image_reconstructed = input_image_reconstructed[idx]
    #         optimizer_re = torch.optim.SGD([image_reconstructed], lr=0.0001)
    #         y_pred = model_f(image_reconstructed).view(-1, 2048)
    #         loss1 = decode_loss_func(y_pred, y).mean()
    #         optimizer_re.zero_grad()
    #         loss1.backward()
    #         optimizer_re.step()
    #         loss.append(loss1)
    #         if i == 0:
    #             input_image_reconstructed.append(image_reconstructed)
    #         else:
    #             input_image_reconstructed[idx] = image_reconstructed
    #     print("loss is", sum(loss)/len(loss))
            
    # # 对比图片
    # for x, (_, _, z) in zip(input_image_reconstructed, feature_dataloader_decode):
    #     x, z = x.to(args.device).float(), z.to(args.device).float()
    #     for i in range(3):
    #         image_array = z[i].permute(1, 2, 0).detach().cpu().numpy()
    #         image_array = (image_array * 255).astype(np.uint8)
    #         image = Image.fromarray(image_array)
    #         image.show()
    #         image_array = x[i].permute(1, 2, 0).detach().cpu().numpy()
    #         image_array = (image_array * 255).astype(np.uint8)
    #         image = Image.fromarray(image_array)
    #         image.show()
    
    # def update_weights_ours_debug2(self, round, idx,  model, discriminator, train_loader, global_proto, gate_model, global_classifier):
    #     # Set mode to train model
    #     for i in range(self.args.num_users):
    #         if i == idx:
    #             model[i].train()
    #         else:
    #             model[i].eval()
    #     gate_model.eval()
    #     features = []
    #     optimizer = torch.optim.SGD(
    #         model[idx].parameters(),
    #         lr=self.args.lr
    #     )
    #     global_classifier.eval()
    #     loss_mse = nn.MSELoss()
    #     criterion_CL = ConLoss(temperature=0.07)
    #     # Set optimizer for the local updates
    #     for step in range(self.args.wk_iters):
    #         train_iter = iter(train_loader)
    #         for batch_idx in range(len(train_iter)):
    #             images, labels = next(train_iter)
    #             images, labels = images.to(self.device).float(), labels.to(self.device).long()
    #             # 保留预测结果
    #             predict_list = []
    #             for client_idx in range(self.args.num_users):
    #                 if client_idx == idx:
    #                     rep, y_pred = model[client_idx](images)
    #                     p = gate_model(rep)
    #                     p_softmax = F.softmax(p, dim=-1)
    #                 else:
    #                     rep_, y_pred = model[client_idx](images)
    #                 predict_list.append(y_pred)
    #             probabilities = p_softmax.permute(1, 0).unsqueeze(-1)  # 变为 [4, 32, 1]
    #             # 按概率加权 predictions，并在第 0 维求和
    #             weighted_predictions = (torch.stack(predict_list, dim=0) * probabilities).sum(dim=0)  # 结果为 [32, 10]
    #             CE_loss = self.criterion_CE(weighted_predictions, labels).mean()
    #             # compute regularized loss term
    #             if len(global_proto) == 0:
    #                 loss1 = 0
    #             else:
    #                 num_classes = len(global_proto)
    #                 proto_dim = global_proto[0].shape[0]
    #                 global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
    #                 # 填充 global_proto_matrix，每一行对应一个类别的原型向量
    #                 for label, proto in global_proto.items():
    #                     global_proto_matrix[label] = proto
    #                 rep_normalized = F.normalize(rep, dim=1)  # [batch_size, d]
    #                 global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)  # [num_classes, d]
    #                 C_y = global_proto_matrix_normalized[labels]  # 选取对应标签的原型向量
    #                 # 分子部分：exp(z_x · C(y) / τ)
    #                 numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
    #                 # 分母部分：对所有可能的类别 A(y) 求和
    #                 denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
    #                 # 损失计算
    #                 loss1 = -torch.log(numerator / denominator).mean()
                    
    #             client_index = discriminator(rep)
    #             client_index_softmax = F.log_softmax(client_index, dim=-1)
    #             target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
    #                 self.device
    #             )
    #             target_index_softmax = F.softmax(target_index, dim=-1)
    #             kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
    #             kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)           
    #             loss = CE_loss + 0.5 * loss1 + 0.5 * kl_loss
    #             # loss = CE_loss + self.args.adcol_mu * kl_loss
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #     # save features
    #     labels_ = []
    #     train_iter = iter(train_loader)
    #     for batch_idx in range(len(train_iter)):
    #         images, labels = next(train_iter)
    #         images, labels = images.to(self.device).float(), labels.to(self.device).long()
    #         rep, logits = model[idx](images)
    #         features.append(rep.detach().clone().cpu())
    #         labels_.append(labels.detach().clone().cpu())
    #     features = torch.cat(features, dim=0).to(self.args.device)
    #     labels_ = torch.cat(labels_, dim=0)
    #     return copy.deepcopy(model[idx]), [features, labels_]    

    # def update_weights_ours_debug3(self, round, idx,  model, discriminator, train_loader, global_proto, gate_model, global_classifier):
    #     # Set mode to train model
    #     for i in range(self.args.num_users):
    #         if i == idx:
    #             model[i].train()
    #         else:
    #             model[i].eval()
    #     gate_model.eval()
    #     features = []
    #     optimizer = torch.optim.SGD(
    #         model[idx].parameters(),
    #         lr=self.args.lr
    #     )
    #     global_classifier.eval()
    #     loss_mse = nn.MSELoss()
    #     criterion_CL = ConLoss(temperature=0.07)
    #     # Set optimizer for the local updates
    #     for step in range(self.args.wk_iters):
    #         train_iter = iter(train_loader)
    #         for batch_idx in range(len(train_iter)):
    #             images, labels = next(train_iter)
    #             images, labels = images.to(self.device).float(), labels.to(self.device).long()
    #             # 保留预测结果
    #             predict_list = []
    #             for client_idx in range(self.args.num_users):
    #                 if client_idx == idx:
    #                     rep, y_pred = model[client_idx](images)
    #                     p = gate_model(rep)
    #                     p_softmax = F.softmax(p, dim=-1)
    #                 else:
    #                     rep_, y_pred = model[client_idx](images)
    #                 predict_list.append(y_pred)
    #             probabilities = p_softmax.permute(1, 0).unsqueeze(-1)  # 变为 [4, 32, 1]
    #             # 按概率加权 predictions，并在第 0 维求和
    #             weighted_predictions = (torch.stack(predict_list, dim=0) * probabilities).sum(dim=0)  # 结果为 [32, 10]
    #             pred_global = global_classifier(rep)
    #             predictions = 0.9 * weighted_predictions + 0.1 * pred_global
    #             CE_loss = self.criterion_CE(predictions, labels).mean()
    #             # compute regularized loss term
    #             if len(global_proto) == 0:
    #                 loss1 = 0
    #             else:
    #                 num_classes = len(global_proto)
    #                 proto_dim = global_proto[0].shape[0]
    #                 global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
    #                 # 填充 global_proto_matrix，每一行对应一个类别的原型向量
    #                 for label, proto in global_proto.items():
    #                     global_proto_matrix[label] = proto
    #                 rep_normalized = F.normalize(rep, dim=1)  # [batch_size, d]
    #                 global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)  # [num_classes, d]
    #                 C_y = global_proto_matrix_normalized[labels]  # 选取对应标签的原型向量
    #                 # 分子部分：exp(z_x · C(y) / τ)
    #                 numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
    #                 # 分母部分：对所有可能的类别 A(y) 求和
    #                 denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
    #                 # 损失计算
    #                 loss1 = -torch.log(numerator / denominator).mean()
                    
    #             client_index = discriminator(rep)
    #             client_index_softmax = F.log_softmax(client_index, dim=-1)
    #             target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
    #                 self.device
    #             )
    #             target_index_softmax = F.softmax(target_index, dim=-1)
    #             kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
    #             kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)           
    #             loss = CE_loss + 0.5 * loss1 + 0.5 * kl_loss
    #             # loss = CE_loss + self.args.adcol_mu * kl_loss
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #     # save features
    #     labels_ = []
    #     train_iter = iter(train_loader)
    #     for batch_idx in range(len(train_iter)):
    #         images, labels = next(train_iter)
    #         images, labels = images.to(self.device).float(), labels.to(self.device).long()
    #         rep, logits = model[idx](images)
    #         features.append(rep.detach().clone().cpu())
    #         labels_.append(labels.detach().clone().cpu())
    #     features = torch.cat(features, dim=0).to(self.args.device)
    #     labels_ = torch.cat(labels_, dim=0)
    #     return copy.deepcopy(model[idx]), [features, labels_]    