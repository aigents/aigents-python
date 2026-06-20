python ./aigents-gym/breakout_eval2.py -cs=2 -ss=0.9 -tu=0 -s=41 -mg=20 | grep -E "scores =|stepss =|livess =|states =" > out20.txt
diff out20.txt out20_base.txt

