buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_beauty --data_dir=amazon --train_dir=beauty_ts_group_negpool_bucket_base --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model SASRec --eval_every_n_epoch=1
buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_beauty --data_dir=amazon --train_dir=beauty_ts_group_negpool_bucket_ihm --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model IHM --eval_every_n_epoch=1
buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_beauty --data_dir=amazon --train_dir=beauty_ts_group_negpool_bucket_cdn --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model SASRec --eval_every_n_epoch=1 --gamma=2.0
buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_beauty --data_dir=amazon --train_dir=beauty_ts_group_negpool_bucket_ihmcdn --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model IHM --eval_every_n_epoch=1 --gamma=2.0

buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_books_sampled --data_dir=amazon --train_dir=books_ts_group_negpool_bucket_base --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model SASRec --eval_every_n_epoch=1
buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_books_sampled --data_dir=amazon --train_dir=books_ts_group_negpool_bucket_ihm --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model IHM --eval_every_n_epoch=1
buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_books_sampled --data_dir=amazon --train_dir=books_ts_group_negpool_bucket_cdn --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model SASRec --eval_every_n_epoch=1 --gamma=2.0
buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_books_sampled --data_dir=amazon --train_dir=books_ts_group_negpool_bucket_ihmcdn --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model IHM --eval_every_n_epoch=1 --gamma=2.0

buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_games --data_dir=amazon --train_dir=games_ts_group_negpool_bucket_base --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model SASRec --eval_every_n_epoch=1
buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_games --data_dir=amazon --train_dir=games_ts_group_negpool_bucket_ihm --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model IHM --eval_every_n_epoch=1
buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_games --data_dir=amazon --train_dir=games_ts_group_negpool_bucket_cdn --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model SASRec --eval_every_n_epoch=1 --gamma=2.0
buck run @mode/opt //sasrec:main_ihm -- --dataset=amazon_games --data_dir=amazon --train_dir=games_ts_group_negpool_bucket_ihmcdn --maxlen=200 --dropout_rate=0.2 --num_epochs 10 --model IHM --eval_every_n_epoch=1 --gamma=2.0

buck run @mode/opt //sasrec:main_ihm -- --dataset=ml-1m_full --data_dir=ml-1m --train_dir=ml1m_ts_group_negpool_bucket_base --maxlen=200 --dropout_rate=0.2 --num_epochs 100 --model SASRec --eval_every_n_epoch=10
buck run @mode/opt //sasrec:main_ihm -- --dataset=ml-1m_full --data_dir=ml-1m --train_dir=ml1m_ts_group_negpool_bucket_ihm --maxlen=200 --dropout_rate=0.2 --num_epochs 100 --model IHM --eval_every_n_epoch=10
buck run @mode/opt //sasrec:main_ihm -- --dataset=ml-1m_full --data_dir=ml-1m --train_dir=ml1m_ts_group_negpool_bucket_cdn --maxlen=200 --dropout_rate=0.2 --num_epochs 100 --model SASRec --eval_every_n_epoch=10 --gamma=2.0
buck run @mode/opt //sasrec:main_ihm -- --dataset=ml-1m_full --data_dir=ml-1m --train_dir=ml1m_ts_group_negpool_bucket_ihmcdn --maxlen=200 --dropout_rate=0.2 --num_epochs 100 --model IHM --eval_every_n_epoch=10 --gamma=2.0
