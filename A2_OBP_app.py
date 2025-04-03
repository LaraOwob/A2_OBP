import streamlit as st
import graphviz
import random
import math
import pyomo.environ as pe
import pyomo.opt as po

def calculate_fraction_uptime(stat_dist,min_components):
    return sum(stat_dist[min_components:])

def from_Q_to_P(Qmatrix):
    max_value =1/abs( min(min(row) for row in Qmatrix))
    matrixP =[[max_value * val for val in row] for row in Qmatrix]
    for i in range(len(matrixP)):
        matrixP[i][i] += 1   
    return matrixP

def calculate_stationary(P, iterations,warm):
    state = [1/len(P)]*len(P)
    if not warm:
        state = [1/(len(P)-1)]*len(P)
        state[0] = 0
    data = [state] 
    for _ in range(iterations):
        state = [sum(P[j][i]*state[j] for j in range(len(P))) for i in range(len(P))]
        data.append(state)
    return state, data  
    
def uptime(failure_rate,repair_rate,warm, no_components,min_components,no_repairman):
    Q = [[0] * (no_components+1) for _ in range(no_components+1)]
    for i in range(no_components+1):
        if i >0:
            if warm:
                Q[i][i - 1] = failure_rate*i
            else:
                if i >= min_components:
                    Q[i][i - 1] = failure_rate*min_components
        if i< no_components:
            Q[i][(i + 1)] = repair_rate*min(no_components - i,no_repairman)
        Q[i][i] = - sum(Q[i])    
    return Q

def calculate_uptime(failure_rate,repair_rate,no_components,min_components,no_repairman,standby_mode):
    Q_matrix = uptime(failure_rate,repair_rate,standby_mode,no_components,min_components,no_repairman)
    P_matrix= from_Q_to_P(Q_matrix)
    stationary_dist,data = calculate_stationary(P_matrix,300,standby_mode)  
    fraction_uptime = calculate_fraction_uptime(stationary_dist,min_components)
    return fraction_uptime


def cost_n(no_components, component_unit):
    cost_component = no_components*component_unit
    return cost_component

def cost_s(no_repairman, repairman_unit):
    cost_repairman = no_repairman*repairman_unit
    return cost_repairman

def cost_d(uptime_fraction,  downtime_unit):
    downtime = (1 - uptime_fraction)
    cost_downtime = downtime * downtime_unit
    return cost_downtime

def calculate_total_cost(no_components,component_unit,no_repairman, repairman_unit,uptime_fraction,  downtime_unit):
    total_cost = cost_n(no_components, component_unit) + cost_s(no_repairman, repairman_unit) + cost_d(uptime_fraction,  downtime_unit)
    return total_cost


def initial_start(min_components):
    no_components = random.randint(min_components, min_components * 3)
    no_repairman = random.randint(1, no_components)
    return no_components, no_repairman

def make_downtime_set(failure_rate,repair_rate,max_components,min_components,standby_mode,downtime_cost):
    downtime_set = {}
    for comp in range(1, max_components + 1):
        for rep in range(1, max_components + 1):
            if comp >= min_components:
                if rep <= comp:
                    uptime_fraction = calculate_uptime(failure_rate, repair_rate, comp, min_components, rep, standby_mode)
                    downtime_set[(comp, rep)] =uptime_fraction 
    return downtime_set

def maxSetSolver(max_components,min_components, component_cost, repair_cost,downtime_cost,failure,repair,iswarm):
    max_repairs = max_components
    u = make_downtime_set(failure,repair,max_components,min_components,iswarm,downtime_cost)
    #Sets and decision variables
    model = pe.ConcreteModel()
    model.C = pe.RangeSet(1,max_components)
    model.R = pe.RangeSet(1,max_repairs)
    model.x = pe.Var(model.C, domain = pe.Binary)
    model.y = pe.Var(model.R, domain = pe.Binary)
    model.z = pe.Var(range(max_components + 1), domain = pe.Binary) #binary for component total
    model.w = pe.Var(range(max_repairs + 1), domain = pe.Binary) #binary for repair total
    
    #Constraints
    model.minComponents = pe.ConstraintList()
    model.minComponents.add(sum(model.x[c] for c in model.C)>= min_components)

    model.minrepair = pe.ConstraintList()
    model.minrepair.add(sum(model.y[r] for r in model.R)>=1)

    model.repairComponents = pe.ConstraintList()
    model.repairComponents.add(sum(model.y[r] for r in model.R) <= sum(model.x[c] for c in model.C))

    model.singleComponent = pe.ConstraintList()
    model.singleComponent.add(sum(model.z[c] for c in range(max_components + 1)) == 1)  #only 1 z[c] should be a 1 (total components)

    model.singleRepair = pe.ConstraintList()
    model.singleRepair.add(sum(model.w[r] for r in range(max_repairs + 1)) == 1) #only 1 w[r] should be 1 (total repairman)

    model.totalComponent = pe.Expression(expr=sum(model.x[c] for c in model.C)) #total number components selected
    model.totalRepair = pe.Expression(expr=sum(model.y[r] for r in model.R)) #total number repairman selected

    model.z_lb = pe.ConstraintList()
    model.z_ub = pe.ConstraintList()
    model.w_lb = pe.ConstraintList()
    model.w_ub = pe.ConstraintList()
    #link z en w met big M

    for c in range(max_components + 1): 
        model.z_lb.add(model.totalComponent - c >= -max_components * (1-model.z[c]))
        model.z_ub.add(model.totalComponent - c <= max_components * (1-model.z[c]))
        
    for r in range(max_repairs + 1):  
        model.w_lb.add(model.totalRepair - r >= -max_repairs * (1-model.w[r]))
        model.w_ub.add(model.totalRepair - r <= max_repairs * (1-model.w[r]))
        
    model.zw = pe.Var([(c,r) for c in range(1, max_components + 1) for r in range(1, max_repairs + 1) if (c,r) in u], domain = pe.Binary)

    model.zwConstraints = pe.ConstraintList()
    for c in range(1, max_components + 1):
        for r in range(1, max_repairs + 1):
            if (c, r) in u:
                model.zwConstraints.add(model.zw[(c, r)] <= model.z[c])
                model.zwConstraints.add(model.zw[(c, r)] <= model.w[r])
                model.zwConstraints.add(model.zw[(c, r)] >= model.z[c] + model.w[r] - 1)

    obj_expr = sum(model.x[c] *component_cost for c in model.C) + sum(model.y[r] *repair_cost  for r in model.R) + sum(cost_d(u[(c,r)], downtime_cost) * model.zw[(c,r)] for c,r in model.zw)
    model.obj = pe.Objective(expr = obj_expr , sense = pe.minimize)
    return model,u
    


def runModelMaxSet(model,u):
    solver = po.SolverFactory('gurobi')
    solver.solve(model, tee =True)
    optimal_components  = sum(pe.value(model.x[c]) for c in model.C)
    optimal_repairman  = sum(pe.value(model.y[r]) for r in model.R)
    optimal_costs  =  pe.value(model.obj)
    optimal_uptime_fraction = u[optimal_components,optimal_repairman]
    return optimal_components, optimal_repairman, optimal_costs,optimal_uptime_fraction

def infinitySolver(min_components,failure,repair,iswarm,component_cost,repairman_cost,downtime_cost,initial_temp,iterations_per_temp):
    x, y = initial_start(min_components)
    best_x, best_y = x, y 
    best_uptime = calculate_uptime(failure, repair, best_x, min_components, best_y, iswarm)
    best_cost = calculate_total_cost(best_x, component_cost, best_y, repairman_cost, best_uptime, downtime_cost)
    no_improvement_cycles = 0  
    max_no_improvement_cycles = 3  
    jumpsize = 0.5
    min_temp = 1
    current_temp = initial_temp
    cooling_rate = 0.95
    while current_temp > min_temp:
        improved = False  
        for _ in range(iterations_per_temp):
            x_new = max(min_components, x + random.choice([-6,-5,-4,-3, -2, -1,0, 1, 2, 3,4,5,6]))
            y_new = max(1, min(x_new, y + random.choice([-6,-5,-4,-3,-2, -1,0, 1, 2,3,4,5,6])))

            uptime_old = calculate_uptime(failure, repair, x, min_components, y, iswarm)
            uptime_new = calculate_uptime(failure, repair, x_new, min_components, y_new, iswarm)

            cost_old = calculate_total_cost(x, component_cost, y, repairman_cost, uptime_old, downtime_cost)
            cost_new = calculate_total_cost(x_new, component_cost, y_new, repairman_cost, uptime_new, downtime_cost)

            if cost_new < cost_old:
                x, y = x_new, y_new 
                uptime_old = uptime_new
            else:
                probability = math.exp((cost_old - cost_new) / (current_temp * jumpsize)) 
                if random.random() < probability:
                    x, y = x_new, y_new  
                    uptime_old = uptime_new

            cost = calculate_total_cost(x, component_cost, y, repairman_cost, uptime_old, downtime_cost)
            if cost < best_cost:
                best_x, best_y = x, y
                best_uptime = uptime_old
                best_cost = cost    
                improved = True 

        if improved:
            no_improvement_cycles = 0 
            jumpsize =0.5
        else:
            no_improvement_cycles += 1 

        if no_improvement_cycles >= max_no_improvement_cycles:
            jumpsize = max(0.2,jumpsize - 0.1)
            no_improvement_cycles = 0  
        current_temp *= cooling_rate

    return best_x, best_y, best_cost,best_uptime



def show_birth_death_diagram_custom(max_components,no_repairman,standby,minimum):
    st.subheader("Birth-Death Process Visualization")
    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='circle', color='red', fontcolor='black', style='bold')
    dot.attr('edge', color='red')
    for i in range(max_components+1):
        dot.node(str(i))
        if i >0:
            if standby:
                failure_rate= i
                dot.edge(str(i), str(i-1), label=str(failure_rate)+ 'μ')
            else:
                if i >= minimum:
                    failure_rate = minimum
                    dot.edge(str(i), str(i-1), label=str(failure_rate)+ 'μ')
        if i< max_components:
            repair_rate  = min(max_components - i,no_repairman)
            dot.edge(str(i), str(i+1), label=str(repair_rate)+'γ')
            
    st.graphviz_chart(dot)

def setup_streamlit_ui():
    st.set_page_config(
        page_title="Optimal search for k out of n maintenance system",
        page_icon="🛠️",
        layout="wide"
    )

def show_user_guide():
    st.markdown("## ℹ️ User Guide")
    st.markdown("""
    This tool calculates the uptime fraction or optimizes the number of components and repairmen in a *k-out-of-n maintenance system*.

    ### General Constraints
    - All input values must be greater than 0
    - Failure and repair rates ≥ 0.01
    - Costs (component, repairman, downtime) ≥ 0
    - Minimal components must be ≤ total number of components
    - Number of repairmen must be ≤ number of components

    ### Interface Behavior
    - You can only run Exercise A or Exercise B, not both at the same time
    - When values are invalid, warnings will appear automatically
    - If you reduce the number of components, the number of repairmen will adjust accordingly
    """)
    st.info("Make sure all values follow the constraints to avoid errors when running the model.")
    
def birth_death_process():
    st.markdown("### Understanding the Birth-Death Process")
    st.markdown("""
    This example of a Birth-Death Process shows how the system transitions between states of failed components in a k-out-of-n maintenance system.

    - **μ**: Repair rate (transitions to the left)
    - **γ**: Failure rate (transitions to the right)
    - Failure rates depend on active components: 2γ when ≥ 2 working, γ when only 1
    """)
    
def explanationA():
    st.markdown("""
    This exercise calculates the fraction of time the system is operational in a k-out-of-n maintenance system, based on the provided reliability parameters. The result is the uptime fraction, i.e the probability that at least k-out-of-n components are functioning at any given time.
    """)


def explanationB():
    st.markdown("""
        Find the **optimal number of components and repairmen** based on several input features, with **two different methods**:

        1. **MaxSet Solver (Gurobi)**: This method provides **exact optimization** and finds the optimal configuration within a defined solution set, as this solver operates within a **predefined set** of maximal components.
        
        2. **Infinity Solver (Simulated Annealing)**: This is a **approximation method** that uses simulated annealing in a large solution set without needing a predefined set of components. While this method may not guarantee the exact optimal result, it offers quicker computations.
        """)
    
    
def setup_sidebar():
    with st.sidebar:
        st.title("Maintenance System")
        show_user_guide()
        st.title("Choose Exercise")
        choice = st.selectbox("Select which exercise to work on:", options=["-", "Exercise A", "Exercise B"], key="exercise_choice")

        if choice == "Exercise A":
            input = {}
            st.session_state["input"] = input
            st.title("Part A: Uptime Calculation")
            input['failure'] = st.number_input("Failure rate", min_value=0.01, value=0.5)
            input['repair'] = st.number_input("Repair rate", min_value=0.01, value=0.8)
            input['warm or cold'] = st.toggle("Standby Mode: ON = Warm, OFF = Cold", value=False, help="Use toggle to choose between Warm or Cold standby")
            input['minimal components'] = st.number_input("Minimum components to run", min_value=1, value=2, step=1)
            input['Number of Components'] = st.number_input("Number of components", min_value=input['minimal components'], value=input['minimal components'], step=1)
            input['Number of Repairman'] = st.number_input("Number of repairmen", min_value=1, max_value=input['Number of Components'], value=input['Number of Components'], step=1)
            st.session_state["Input phase A"] = input
            run_a = st.button("Run Exercise A")
            return [choice], run_a

        elif choice == "Exercise B":
            input = {}
            st.session_state["input"] = input
            st.title("Part B: Optimization")
            input['failure'] = st.number_input("Failure rate", min_value=0.01, value=0.5)
            input['repair'] = st.number_input("Repair rate", min_value=0.01, value=0.8)
            input['warm or cold'] = st.toggle("Standby Mode: ON = Warm, OFF = Cold", value=False, help="Use toggle to choose between Warm or Cold standby")
            input['minimal components'] = st.number_input("Minimum components to run", min_value=1, value=2, step=1)
            input['Component Cost'] = st.number_input("Cost per component", min_value=0.0, value=3.0)
            input['Repair cost'] = st.number_input("Cost per repairman", min_value=0.0, value=10.0)
            input['Down time cost'] = st.number_input("Downtime cost", min_value=0.0, value=100.0)
            solver_choice = st.selectbox("Select Solver", ["Infinity Solver", "Max Set Solver"])

            if solver_choice == "Max Set Solver":
                input['Maximum components'] = st.number_input("Maximum components to search", min_value=input['minimal components'], value=input['minimal components'] * 3, step=1)
            if solver_choice == "Infinity Solver":
                input['Run time'] = st.number_input("Running time to find the optimal solution", min_value=20, value=30)

            st.session_state["Input phase B"] = input
            run_b = st.button("Run Exercise B")
            return [choice, solver_choice], run_b

        return [], False

def main():
    setup_streamlit_ui()
    col1, col2 = st.columns([0.1, 2.5], gap="small")
    with col2:
        col2_left, col2_right = st.columns([1, 1])
        with col2_left:
            st.title("Maintenance System Modelling")
            st.markdown("""
                Welcome to the Maintenance system optimization tool.  
                This tool allows you to:
                1. Calculate uptime based on system parameters  
                2. Find the optimal number of components and repairmen to minimize downtime cost
            """)
        with col2_right:
            st.text("")  
            st.text("")  
            birth_death_process()
    with col1:
        choices, run_pressed = setup_sidebar()
    if not choices:
        return
    choicePhase = choices[0]
    solver_choice = choices[1] if len(choices) > 1 else None
    if choicePhase == "Exercise A":
        st.header("Exercise A: Uptime Calculator")
        explanationA()
        phase_A_input = st.session_state["Input phase A"]

        if run_pressed:
            with st.spinner("Solving uptime..."):
                uptime = calculate_uptime(
                    phase_A_input['failure'],
                    phase_A_input['repair'],
                    phase_A_input['Number of Components'],
                    phase_A_input['minimal components'],
                    phase_A_input['Number of Repairman'],
                    phase_A_input['warm or cold']
                )

            results = {"Phase": "A", "Uptime fraction": uptime}
            st.session_state["results"] = results
            st.success("Uptime calculation complete!")
            st.balloons()
            st.markdown("### Summary of Your Inputs")
            st.table({
                "Failure Rate": [f"{phase_A_input['failure']:.2f}"],
                "Repair Rate": [f"{phase_A_input['repair']:.2f}"],
                "Standby Mode": ["Warm" if phase_A_input['warm or cold'] else "Cold"],
                "Min Components": [f"{int(phase_A_input['minimal components'])}"],
                "Total Components": [f"{int(phase_A_input['Number of Components'])}"],
                "Repairmen": [f"{int(phase_A_input['Number of Repairman'])}"],
                "Uptime Fraction": [round(uptime, 4)]
            })
            show_birth_death_diagram_custom(int(phase_A_input['Number of Components']),int(phase_A_input['Number of Repairman']),
                                phase_A_input['warm or cold'],int(phase_A_input['minimal components']))

    elif choicePhase == "Exercise B":
        st.header("Exercise B: Optimize Components and Repairmen")
        explanationB()
        phase_B_input = st.session_state["Input phase B"]
        if run_pressed:
            with st.spinner("Optimizing..."):
                if solver_choice == "Max Set Solver":
                    setModel, uptime_set = maxSetSolver(
                        phase_B_input["Maximum components"],
                        phase_B_input['minimal components'],
                        phase_B_input['Component Cost'],
                        phase_B_input['Repair cost'],
                        phase_B_input['Down time cost'],
                        phase_B_input['failure'],
                        phase_B_input['repair'],
                        phase_B_input['warm or cold']
                    )
                    optimal_components, optimal_repairman, optimal_cost, optimal_uptime = runModelMaxSet(setModel, uptime_set)
                else:
                    optimal_components, optimal_repairman, optimal_cost, optimal_uptime = infinitySolver(
                        phase_B_input['minimal components'],
                        phase_B_input['failure'],
                        phase_B_input['repair'],
                        phase_B_input['warm or cold'],
                        phase_B_input['Component Cost'],
                        phase_B_input['Repair cost'],
                        phase_B_input['Down time cost'],
                        phase_B_input['Run time'],
                        iterations_per_temp=10
                    )

            results = {
                "Phase": "B",
                "Solver used": solver_choice,
                "Optimal components": optimal_components,
                "Optimal repairman": optimal_repairman,
                "Optimal costs": optimal_cost,
                "Uptime fraction": optimal_uptime
            }
            st.session_state["results"] = results
            st.success("Optimization complete!")
            st.balloons()
            st.markdown("### Optimization Summary")
            
            st.table({
            "Failure Rate": [f"{phase_B_input['failure']:.2f}"],
            "Repair Rate": [f"{phase_B_input['repair']:.2f}"],
            "Standby Mode": ["Warm" if phase_B_input['warm or cold'] else "Cold"],
            "Min Components": [phase_B_input['minimal components']],
            "Component Cost": [f"€{phase_B_input['Component Cost']:.2f}"],
            "Repairman Cost": [f"€{phase_B_input['Repair cost']:.2f}"],
            "Downtime Cost": [f"€{phase_B_input['Down time cost']:.2f}"],
            "Optimal Components": [f"{int(optimal_components)}"],
            "Optimal Repairmen": [f"{int(optimal_repairman)}"],
            "Total Cost": [f"€{optimal_cost:.2f}"],
            "Uptime Fraction": [f"{optimal_uptime:.4f}"],
            "Solver Used": [solver_choice]
            })
            
            show_birth_death_diagram_custom(int(optimal_components),int(optimal_repairman),
                                            phase_B_input['warm or cold'],int(phase_B_input['minimal components']))
                                

if __name__ == "__main__":
    main()