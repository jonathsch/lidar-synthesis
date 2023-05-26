export CARLA_ROOT=/home/jonathan/CARLA-10/
export WORK_DIR=/home/jonathan/dev/lidar-augmentation

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/src/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/src/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/src/leaderboard/data/training/scenarios/Scenario1/Town01_Scenario1.json
export ROUTES=${WORK_DIR}/src/leaderboard/data/training/routes/Scenario1/Town01_Scenario1.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/transfuser_weather_14_soft_rain_sunset.json
export TEAM_AGENT=${WORK_DIR}/src/agents/team_code_transfuser/submission_agent.py
export TEAM_CONFIG=${WORK_DIR}/model_ckpt/transfuser
export DEBUG_CHALLENGE=1
export RESUME=0
export DATAGEN=0

python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME}
