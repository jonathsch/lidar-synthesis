export CARLA_ROOT=/home/jonathan/CARLA-10/
export WORK_DIR=/home/jonathan/dev/lidar-synthesis

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/lidar_synthesis/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/lidar_synthesis/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/lidar_synthesis/leaderboard/data/training/scenarios/Scenario1/Town01_Scenario1.json
export ROUTES=${WORK_DIR}/lidar_synthesis/leaderboard/data/training/routes/Scenario1/Town01_Scenario1.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/lidar2waypoints.json
export TEAM_AGENT=${WORK_DIR}/lidar_synthesis/agents/lidar2waypoints_agent.py
export TEAM_CONFIG=${WORK_DIR}/model_ckpt/lidar_waypoints
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
