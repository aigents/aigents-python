#python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -s=2 -mg=50000000 -mt=108000 -sm="exp(-d)" -tu=0 -cc=0.0 -r="human"
python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -tu=0 -s=2  -mg=5000000 -mt=1080000 -sm="exp(-d)" > sim_test5/exp1080000_s2ss10lm2sc2cs2tu0er0mc0.txt
python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -tu=0 -s=3  -mg=5000000 -mt=1080000 -sm="exp(-d)" > sim_test5/exp1080000_s3ss10lm2sc2cs2tu0er0mc0.txt
python ./aigents-gym/breakout_eval2.py -cs=2 -ss=1.0 -tu=0 -s=41 -mg=5000000 -mt=1080000 -sm="exp(-d)" > sim_test5/exp1080000_s41ss10lm2sc2cs2tu0er0mc0.txt

