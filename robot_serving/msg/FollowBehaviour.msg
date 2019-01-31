# ----------------------------------

int8 follow_rule
bool with_head
bool with_eyes
string name

# ----------------------------------

# for follow_behaviour
int8 DEFAULT = 0
int8 FOLLOW_FACES = 1
int8 FOLLOW_LINK = 2
int8 FOLLOW_OBJECT = 3
int8 FOLLOW_RANDOM_LOOK = 4

# link names
string LEFT_GRIPPER = /left_gripper
string RIGHT_GRIPPER = /right_gripper