# num_batches = 10
# t0 = time.time()
# for i, (inputs, _) in enumerate(train_loader):
#     inputs = inputs.to(device)
#     lit_module(inputs, mask_ratio=0.0)
#     if i >= num_batches:
#         break
# torch.cuda.synchronize()
# print((time.time() - t0) / num_batches)
