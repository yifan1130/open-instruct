requirements = (Machine ==  "isye-hpc0457.isye.gatech.edu")
universe = vanilla
getenv = true
executable = gsm.sh
notify_user = yyu429@gatech.edu
Log = /home/yyu429/eval_ppl2/$(Cluster).$(process).log
output = /home/yyu429/eval_ppl2/$(Cluster).$(process).out
error = /home/yyu429/eval_ppl2/$(Cluster).$(process).error
notification = error
notification = complete
request_gpus = 1
queue