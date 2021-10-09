flow_size = 100
pkt_size = 200
column_batch_size = 128  # each column will take 128 flows to learn from
learning_batch_size = 16 # each column will learn will batch = 16
base_batch_sizes = [4,8,16,32,64,128]
learning_rate = 1e-3
epochs = 50 
task_epochs = 20
test_labels = ['benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan']
all_labels = ['vectorize_friday/benign', 'attack_bot', 'attack_DDOS',\
            'attack_portscan', 'Benign_Wednesday', 'DOS_SlowHttpTest',\
            'DOS_SlowLoris', 'DOS_Hulk', 'DOS_GoldenEye', 'FTPPatator',\
            'SSHPatator', 'Web_BruteForce', 'Web_XSS']
