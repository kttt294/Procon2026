TERRAIN_PLAIN    = 0
TERRAIN_MOUNTAIN = 1
TERRAIN_LAKE     = 2
TERRAIN_ROAD     = 3

AGENT_PATROL = 0
AGENT_SUPPLY = 1

TRAFFIC_CLEAR     = 0
TRAFFIC_BUSY      = 1
TRAFFIC_CONGESTED = 2

# Step cost: cost to move WHILE ON this terrain (source-based).
# Road cost depends on traffic status.
STEP_COST = {
    TERRAIN_PLAIN:    2,
    TERRAIN_MOUNTAIN: 3,
    TERRAIN_LAKE:     None,   # impassable
    TERRAIN_ROAD:     {
        TRAFFIC_CLEAR:     1,
        TRAFFIC_BUSY:      2,
        TRAFFIC_CONGESTED: 4,
    },
}

# Fuel cost: cost to move WHILE ON this terrain (source-based, patrol only).
FUEL_COST = {
    TERRAIN_PLAIN:    1,
    TERRAIN_MOUNTAIN: 2,
    TERRAIN_ROAD:     2,
}

N_DIRECTIONS = 6

# RL hyperparameters
HIDDEN_DIM    = 256
LR_ACTOR      = 3e-4
LR_CRITIC     = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
ENTROPY_COEF  = 0.01
N_EPOCHS      = 4
BATCH_SIZE    = 64

# Reward weights
RW_NEW_SERIES   = 100.0
RW_DAILY_SERIES =  10.0
RW_UDON         =   1.0
RW_FUEL_EMPTY   = -10.0
RW_WASTED_STEP  =  -0.05

# Potential-based reward shaping: phi(s) = -POTENTIAL_SCALE * mean_min_hex_dist_to_uncollected_spot
# Shaped reward = r + gamma * phi(s') - phi(s)
# Scale 10 → getting 1 hex closer ≈ 10 shaped reward units (comparable to RW_UDON)
POTENTIAL_SCALE = 10.0
