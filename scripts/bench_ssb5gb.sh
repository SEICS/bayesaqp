# # declare -a ssb-5gb_queries=("ssb-5gb_q0" "ssb-5gb_q1" "ssb-5gb_q2"
# # 							"ssb-5gb_q3" "ssb-5gb_q4" "ssb-5gb_q5"
# # 							"ssb-5gb_q6" "ssb-5gb_q7" "ssb-5gb_q8"
# # 							"ssb-5gb_q9" "ssb-5gb_q10" "ssb-5gb_q11")
# CMD="target/native-image/bayestensor" # macosx

CMD="target/native-image/bayestensor" # ubuntu

# # for q in "${ssb-5gb_queries[@]}"
# # do
# # 	hyperfine --warmup 3 --runs 5 "$CMD ../aqp/flights/codegen/$q bt"
# # done
# END = 11
# for ((i=0;i<=11;i++)); #"${ssb-5gb_queries[@]}" # $(seq 0 $END);
# do 
# 	hyperfine --warmup 3 --runs 5 "$CMD progs/aqp/ssb/codegen/avg_gb/ssb-5gb_q$i bt"
# 	# hyperfine --warmup 3 --runs 5 "$CMD progs/aqp/ssb/codegen/sum_gb/ssb-5gb_q$i bt"
# done

# declare -a ssb5gb_queries=("ssb-5gb_q0" "ssb-5gb_q1" "ssb-5gb_q2"
# 							"ssb-5gb_q3" "ssb-5gb_q4" "ssb-5gb_q5"
# 							"ssb-5gb_q6" "ssb-5gb_q7" "ssb-5gb_q8"
# 							"ssb-5gb_q9_A" "ssb-5gb_q9" 
# 							"ssb-5gb_q10_A" "ssb-5gb_q10" 
# 							"ssb-5gb_q11_A" "ssb-5gb_q11")
bin=1000
declare -a ssb5gb_queries=(
							"ssb-5gb_q3" "ssb-5gb_q4" "ssb-5gb_q5"
							"ssb-5gb_q6" "ssb-5gb_q7" "ssb-5gb_q8"
							"ssb-5gb_q9") 
							#  "ssb-5gb_q10"  "ssb-5gb_q11"
							# "ssb-5gb_q12")

for q in "${ssb5gb_queries[@]}"
do
	hyperfine --warmup 4 --runs 5 "$CMD progs/aqp/ssb/codegen/aqp_not_sampled/sf5/bin$bin/$q bt"
done
