# declare -a flights_queries=("flights_q0" "flights_q1" "flights_q2" "flights_q3"
# 							"flights_q4" "flights_q5" "flights_q6" "flights_q7" 
# 							"flights_q8" "flights_q9" "flights_q10" "flights_q11")
CMD="target/native-image/bayestensor"
declare -a flights_queries=("flights_q0" "flights_q1" "flights_q2" "flights_q3" 
							"flights_q4" "flights_q5" "flights_q6" "flights_q7" 
							"flights_q8" "flights_q9" "flights_q10" "flights_q11")
# declare -a flights_queries=("flights_q0")
bin=1500
for q in "${flights_queries[@]}"
do
	# hyperfine --warmup 4 --runs 5 "$CMD progs/aqp/flights/codegen/1B/bin$bin/$q bt" # if numpy gpu
	hyperfine --warmup 4 --runs 5 "$CMD progs/aqp/flights/codegen/1B/bin$bin/$q bt torch" # if torch gpu
done
