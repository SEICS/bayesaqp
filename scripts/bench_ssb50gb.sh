CMD="target/native-image/bayestensor" 

bin=1000
declare -a ssb50gb_queries=("ssb-50gb_q0" "ssb-50gb_q1" "ssb-50gb_q2"
							"ssb-50gb_q3" "ssb-50gb_q4" "ssb-50gb_q5"
							"ssb-50gb_q6" "ssb-50gb_q7" "ssb-50gb_q8"
							"ssb-50gb_q9" "ssb-50gb_q10"  "ssb-50gb_q11"
							"ssb-50gb_q12")

for q in "${ssb50gb_queries[@]}"
do
	hyperfine --warmup 4 --runs 5 "$CMD progs/aqp/ssb/codegen/aqp_sampled/sf50/bin$bin/$q bt numpy"
done

# for q in "${ssb50gb_queries[@]}"
# do
# 	hyperfine --warmup 4 --runs 5 "$CMD progs/aqp/ssb/codegen/aqp_sampled/sf50/bin$bin/$q bt torch"
# done
