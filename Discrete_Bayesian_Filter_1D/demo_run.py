# StrikePS

def main():
    """ This is a demo file to use the DiscreteFilter class in 1D world. """
    
    import numpy as np

    from DiscreteFilter1D import DiscreteFilter1D as DF

    from plot_utils import plot_belief_evolution as pbe

    # define 1d world.
    world_size = 20
    x = np.zeros(world_size)
    index = int(input(f"Enter the initial position of the robot (1,{world_size}): "))
    x[index-1] = 1
    print(f"Initial position of the robot: {index}")
    print(x)
    print("")

    #define intial set of beliefs
    bel = np.zeros(world_size)
    bel[index-1] = 1

    # set the tower postions
    towers = [3,16]

    # define the robot
    rob = DF(x,bel,towers)
    
    # define the number of time steps
    num_steps = 20
    
    #define the commands array
    commands = np.zeros(num_steps)
    
    # Uncomment the following lines to take user input for commands
    # for i in range(num_steps):
    #     commands[i] = int(input(f"Enter command for step {i+1} (0 for no move, 1 for forward, 2 for backward): "))
    #     if commands[i] not in [0, 1, 2]:
    #         print("Invalid command. Please enter 0, 1, or 2.")
    #         commands[i] = int(input(f"Enter command for step {i+1} (0 for no move, 1 for forward, 2 for backward): "))
    
    # commands for testing. only 3 backward moves at steps 10, 11, and 12
    commands = np.ones(num_steps)
    commands[9] = commands[10] = commands[11] = 2
    
    
    # define the data type array
    # 0 for action data, 1 for observation data
    data_type = np.hstack(np.zeros(num_steps))  # implementing only action data for now.
    
    # exemplar mixed data type for testing
    # data_type = np.hstack((0,0,0,0,1,1,0,1,0,0,1,1,1,0,1,1,0,0,0,1,np.zeros(num_steps)))
    
    # stores the complete beliefs update. 
    bel_hist = [rob.bel.copy()]
    
    # run a loop for the number of time steps
    for i in range(num_steps):
        rob.x, rob.bel = rob.update(commands[i], data_type[i])
        bel_hist.append(rob.bel.copy())
    
    # print the real final position of the robot (by real we mean if it had moved without noise)
    dex = np.where(rob.x == 1)[0][0]
    print(f"From Noise Free Updates, Final Position of Robot: {dex+1}")
    print(rob.x) # prints the array with 1 at the final position of the robot.
    print("")

    # print the beliefs of the robot.
    print("Robot beliefs:")
    print(rob.bel)
    print(f"print robot beliefs sum: {np.sum(rob.bel)}")
    print("")
    
    # print the max belief value of the robot
    print("max belief of robot:")
    print(np.max(rob.bel))
    print("")
    
    # taking the assumption that the robot exists at the position with max belief. print the index of that position
    dex = np.where(rob.bel == np.max(rob.bel))[0][0]
    print(f"Max belief position of robot: {dex+1}")
    dex = np.where(rob.bel == np.max(rob.bel))[0][0]
    
    # make a visual plot of the belief evolution
    # rows represent a time step and columns represent the belief at each position in the world.
    pbe(bel_hist)

if __name__ == "__main__":
    main()