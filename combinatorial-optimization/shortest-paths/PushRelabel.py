def PushRelabel(capacities, source, sink, verbose=True):
    if verbose:
        # V is the number of nodes. We use len(capacities) which accounts 
        # for our 1-based indexing (index 0 is ignored).
        V = len(capacities)
        
        # Initialize flow, excess, and height arrays
        flow = [[0] * V for _ in range(V)]
        excess = [0] * V
        height = [0] * V

        # Helper function to calculate residual capacity
        def residual_capacity(u, v):
            return capacities[u][v] - flow[u][v]

        # --- STEP 0: INITIALIZATION ---
        height[source] = V - 1 # V-1 because index 0 is unused in our 1-based graph
        
        print("-" * 60)
        print("Step 0 | Action: Initialize Preflow")
        
        # Push maximum flow to all neighbors of the source
        for v in range(1, V):
            if capacities[source][v] > 0:
                f = capacities[source][v]
                flow[source][v] += f
                flow[v][source] -= f  # Create residual backward edge
                excess[v] += f
                excess[source] -= f
                
        print(f"  Excess (Nodes 2-5): {excess[2:6]}")
        print(f"  Height (Nodes 1-6): {height[1:7]}")
        print("-" * 60)

        step = 1
        
        # --- MAIN LOOP ---
        while True:
            # Rule 1: Find active node with the smallest index
            u = -1
            for i in range(1, V):
                if i != source and i != sink and excess[i] > 0:
                    u = i
                    break
                    
            # If no intermediate node has excess, the algorithm is complete
            if u == -1:
                break

            # Rule 2: Examine neighbors in increasing numerical order
            action_taken = False
            
            # Try to Push
            for v in range(1, V):
                if residual_capacity(u, v) > 0 and height[u] == height[v] + 1:
                    # PUSH
                    delta = min(excess[u], residual_capacity(u, v))
                    flow[u][v] += delta
                    flow[v][u] -= delta  # Update backward residual edge
                    excess[u] -= delta
                    excess[v] += delta
                    
                    print(f"Step {step:<2} | Node {u} | Action: Push ({u} -> {v}) by {delta} units")
                    print(f"  Excess (Nodes 2-5): {excess[2:6]}")
                    print(f"  Height (Nodes 1-6): {height[1:7]}")
                    
                    step += 1
                    action_taken = True
                    break # Perform only one action per step

            # If we couldn't push, we must Relabel
            if not action_taken:
                min_height = float('inf')
                for v in range(1, V):
                    if residual_capacity(u, v) > 0:
                        min_height = min(min_height, height[v])
                
                height[u] = 1 + min_height
                
                print(f"Step {step:<2} | Node {u} | Action: Relabel (New Height: {height[u]})")
                print(f"  Excess (Nodes 2-5): {excess[2:6]}")
                print(f"  Height (Nodes 1-6): {height[1:7]}")
                
                step += 1

        # Calculate final max flow (sum of flow into the sink)
        max_flow = sum(flow[u][sink] for u in range(1, V))
        print("-" * 60)
        print(f"ALGORITHM COMPLETE. Maximum Flow: {max_flow}")
        return max_flow
    
    else:
        # If not verbose, just compute the max flow without printing steps
        V = len(capacities)
        flow = [[0] * V for _ in range(V)]
        excess = [0] * V
        height = [0] * V

        def residual_capacity(u, v):
            return capacities[u][v] - flow[u][v]

        height[source] = V - 1
        
        for v in range(1, V):
            if capacities[source][v] > 0:
                f = capacities[source][v]
                flow[source][v] += f
                flow[v][source] -= f
                excess[v] += f
                excess[source] -= f

        while True:
            u = -1
            for i in range(1, V):
                if i != source and i != sink and excess[i] > 0:
                    u = i
                    break
            
            if u == -1:
                break

            action_taken = False
            
            for v in range(1, V):
                if residual_capacity(u, v) > 0 and height[u] == height[v] + 1:
                    delta = min(excess[u], residual_capacity(u, v))
                    flow[u][v] += delta
                    flow[v][u] -= delta
                    excess[u] -= delta
                    excess[v] += delta
                    
                    action_taken = True
                    break

            if not action_taken:
                min_height = float('inf')
                for v in range(1, V):
                    if residual_capacity(u, v) > 0:
                        min_height = min(min_height, height[v])
                
                height[u] = 1 + min_height

        max_flow = sum(flow[u][sink] for u in range(1, V))
        return max_flow