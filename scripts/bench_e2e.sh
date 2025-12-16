declare -a bns=("cancer" "survey" "alarm" "insurance" "hepar2" "pigs" "water" "munin" "hailfinder")
declare -a baselines=("alarm" "evidence1" "evidence2" "grass" "noisyOr" "twoCoins" "murderMystery")
CMD="target/native-image/bayestensor"

# target/native-image/bayestensor ../aqp/flights/codegen/flights_q0 bt

for bn in "${bns[@]}"
do
	hyperfine --warmup 3 --runs 5 "$CMD $bn bn"
done
for baseline in "${baselines[@]}"
do
	hyperfine --warmup 3 --runs 5 "$CMD $baseline baseline"
done