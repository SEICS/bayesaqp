CMD="target/native-image/bayestensor"
# bin=1000

declare -a join_queries=("count_no_gb" "avg_no_gb" "sum_no_gb"
						   "count_2gb" "avg_2gb" "sum_2gb")

declare -a no_join_queries=("count_no_gb" "avg_no_gb" "sum_no_gb"
						   "count_gb" "avg_gb" "sum_gb")

# for q in "${join_queries[@]}"
# do
# 	hyperfine --warmup 4 --runs 5 "$CMD progs/aqp/ssb/codegen/aqp_not_sampled/sf5/micro/bin1000/q3/join/$q bt"
# done

for q in "${no_join_queries[@]}"
do
	hyperfine --warmup 4 --runs 5 "$CMD progs/aqp/ssb/codegen/aqp_not_sampled/sf5/micro/bin1000/q3/no_join/$q bt"
done