### 2018.7.11
? compare difference between PointNet
	x provider.py
		x add "rotate_perturbation_point_cloud"
		x add "shift_point_cloud"
		x add "random_scale_point_cloud"
	x train.py
		x default model change to dgcnn
		x jittered_data add scale/perturbation/shift
	? models/transform_nets.py
		x point_cloud -> edge_feature
		? is_dist in tf_util.py indicate distributed training for BN
		? tf.reduce_max after conv64-128
	? models/dgcnn.py
		? get_model different
		x do not have the transformation matrix loss
		x main as test file is different
	x tf_util.py
		x add distributed training for BN
		x pairwise_distance of point cloud
		x knn: get knn based on pairwise distance
		x get_edge_feature: construct edge_feature for each point
	? part_seg/train_multi_gpu.py
		x import part_seg_model as model
		? calcualte average_gradients

### 2018.7.17
x structure of code for part_seg/train_multi_gpu 
	(assume batch_size=2, num_points=2048, k=30)
	x set parser / parameters
	x train
		x load place_holder for two GPUs: [2, 2048, 3] 
		x part_seg_model.get_model
			x tf_util.pairwise_distance: [2, 2048, 30, 30], [30,30] denotes the distance
			x tf_util.knn: [2, 2048, 30], [30] denotes the knn index
			x tf_util.get_edge_feature: [2, 2048, 30, 6], [6] = [x_i, x_j - x_i], combine local and global features
			x models/transform_nets.input_transform_net: get [2, 3, 3] transform matrix, apply to point_cloud
				x tf_util.conv2d / tf_util.max_pool2d
				x 3 - conv64 - conv128 - \square=MAX - conv1024 - max - fc512 - fc256 - fc9
			x 3 - [transform3] - [conv64 - conv64 - \square=MAX+MEAN - conv64] - [conv64 - \square=MAX+MEAN - 64] - [conv64 - \square=MAX+MEAN - conv64 - conv1024 - max - concate[features,label] - conv256 - dp0.6 - conv256 - dp0.6 - conv128 - conv50] 
		x part_seg_model.get_loss
		x summary
		x train_one_epoch
		x eval_one_epoch

Notice:
x code can only run when num_gpu==2
x structure of network is different from paper
	x both max and mean for \square
	x \square not always apply to the last layer
	x concatenate all previous output
	x concatenate the category label 	

### 2018.7.18
x worktree under test_nina
    x change provider.py to omit download
    x change train_multi_gpu.py the data loading directory
x no_transform: delete transform network in part_seg_model.py
x paper_architecture: change the network architecture in part_seg_model.py to match the paper
