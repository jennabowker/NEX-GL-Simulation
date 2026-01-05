# Load packages needed to run model
from mesa import Agent as MesaAgent, Model
from mesa.time import StagedActivation 
from mesa.space import MultiGrid
import random
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# Create function of the NEX-GL computational model
def run_simulation(context, mean_kin_m, mean_kin_w, sd_kin_m, sd_kin_w, max_connections, contact_mean, contact_sd, max_steps = 5):

    # Define variables
    data = {
        'step': [],
        'agent_id': [],
        'gender': [],
        'level': [],
        'SC': [],
        'resources': [],
        'RG': [],
        'RCB': [],
        'KC': [],
        'NO': [],
        'PM': [],
        'PW': [],
        'DC': []
    }
    
    # Set global variables
    total_agents = 3000
    max_SC = total_agents * 0.90
    utility_threshold = 0.5

    # Set "world" boundaries (organizational hierarchy)
    world_width = 500
    world_height = 500  

    level_counts = {
        1: round(13000 / 30600 * total_agents),
        2: round(10000 / 30600 * total_agents),
        3: round(5000 / 30600 * total_agents),
        4: round(2000 / 30600 * total_agents),
        5: round(1000 / 30600 * total_agents),
        6: round(500 / 30600 * total_agents),
        7: round(100 / 30600 * total_agents),
        8: 1
    }

    # Function to determine industry type based on context
    def determine_industry_type(context):
        if context < 0.4:
            return "female-dominated"
        elif 0.4 <= context <= 0.6:
            return "egalitarian"
        else:
            return "male-dominated"

    # Define agents (create agentSet in Mesa)
    class CustomAgent(MesaAgent):
        # Initialize agents with relevant variables
        def __init__(self, unique_id, model, level):
            super().__init__(unique_id, model)
            self.level = level
            self.gender = 0 if random.random() < context else 1
            self.age = np.random.poisson(35) if level <= 4 else np.random.poisson(55)
            self.GRE = 0  # Gender Role Expectations
            self.RCB = 0  # Role Consistent Behavior
            self.LA = self.calculate_leadership_aspiration(self.gender, model.industry_type) # Leadership Aspirations
            self.social_contacts = []
            self.KC = 0  # Kin Contacts
            self.max_con = max_connections
            self.SC = 0  # Social Capital
            self.interaction_complete = False
            self.request_complete = False
            self.relationship_assess = False
            self.resources = 0
            self.RS = 0  # Resources Shared
            self.RG = 0 # Resources Given
            self.previous_SC = 0
            self.NO = 0  # Network Openness
            self.PW = 0  # Proportion of Female Contacts
            self.PM = 0  # Proportion of Male Contacts
            self.DC = 0
            self.contact_activated = False  # Contact activation flag

        # Define function to initialize agents
        def setup_agent(self):
            self.generate_KC()
            self.max_con = max_connections - self.KC
            self.assign_gender_role_expectations()
            self.assign_initial_SC()

        # Define function to assign leadership aspirations based on Netchaeva et al (2022)
        def calculate_leadership_aspiration(self, gender, industry):
            if gender == 0:  # Male
                return 1 if random.random() < 0.25 else 0
            else:  # Female
                if industry == "female-dominated":
                    return 1 if random.random() < 0.23525 else 0  # 23.525% for women in female-dominated
                elif industry == "egalitarian":
                    return 1 if random.random() < 0.23125 else 0  # 23.125% for women in egalitarian
                elif industry == "male-dominated":
                    return 1 if random.random() < 0.22725 else 0  # 22.725% for women in male-dominated
                else:
                    return 0  # Default case (should never happen if context is set correctly)

        # Define function to assign gender contacts based on empirical evidence
        def generate_KC(self):
            if self.gender == 0:
                self.KC = round(np.random.normal(mean_kin_m, sd_kin_m))
            else:
                self.KC = round(np.random.normal(mean_kin_w, sd_kin_w))

        # Define function to assign gender role expectations
        def assign_gender_role_expectations(self):
            base_male_GRE = 0.6 # Men report higher raditional beliefs
            base_female_GRE = 0.4 # Women report lower 
            if self.gender == 0: # Male dominated contexts have more traditional norms
                expectations = np.random.normal(base_male_GRE + 0.3 * context, 0.15)
            else:
                expectations = np.random.normal(base_female_GRE + 0.2 * context, 0.15)
            self.GRE = min(max(expectations, 0), 1)

        # Define function to assign initial capital
        def assign_initial_SC(self):
            level_factor = self.level / 8
            gender_factor = np.random.normal(context, 0.10) if self.gender == 0 else np.random.normal(context, 0.05) # CHANGED
            kc_factor = 1 - (self.KC / self.max_con)
            network_factor = (len(self.social_contacts) + (0.1 if self.gender == 1 else 0)) / self.max_con if self.max_con > 0 else 0 # CHANGED
            self.SC = (level_factor + gender_factor + kc_factor + network_factor) / 4
            
            # Ensure total SC does not exceed max_SC to create a SC ceiling
            total_SC = sum([agent.SC for agent in self.model.schedule.agents])
            if total_SC > max_SC:
                scaling_factor = max_SC / total_SC
                self.SC *= scaling_factor
            
            self.resources = self.SC  # Initialize resources as SC

        def initial_contacts(self):
        # Generate the number of initial contacts from a normal distribution and find other agents not in their network
            num_contacts = round(np.random.normal(contact_mean, contact_sd))
            if num_contacts < 1:
                num_contacts = 1
            candidates = [agent for agent in self.model.schedule.agents if agent != self and agent not in self.social_contacts]
            random.shuffle(candidates)  # Shuffle to randomize the candidates

        # Assign contacts up to the number of contacts needed
            assigned_contacts = 0
            while assigned_contacts < num_contacts and candidates:
                candidate = candidates.pop(0)
            # Ensure reciprocal connection: Add the candidate to this agent's contacts and vice versa
                if candidate not in self.social_contacts:
                    self.social_contacts.append(candidate)
                    assigned_contacts += 1
                if self not in candidate.social_contacts:
                    candidate.social_contacts.append(self)

        # Define function to calculate role-consistent behaviors
        def calculate_RCB(self):
            male_contacts = sum(1 for contact in self.social_contacts if contact.gender == 0)
            female_contacts = sum(1 for contact in self.social_contacts if contact.gender == 1)

            # Incorporate the number of same gender ("role-consistent") contacts in network
            total_contacts = male_contacts + female_contacts
            if total_contacts > 0:
                opposite_gender_contacts = female_contacts if self.gender == 0 else male_contacts
                num_contacts_scaler = 1 - (opposite_gender_contacts / total_contacts) if opposite_gender_contacts > 0 else 1
                gre_scaler = self.GRE
                self.RCB = (num_contacts_scaler + gre_scaler) / 3

        # Define function for agent interaction to begin network creation and use
        def interact_with_nearby_agents(self):
            if not self.interaction_complete and len(self.social_contacts) < self.max_con:
                # Find nearby agents within a radius of 3 that are not in the agent's social contacts
                neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=3)
                for neighbor in neighbors: # Assess utility and add if above threshold
                    if neighbor != self and neighbor not in self.social_contacts:
                        neighbor_utility = self.model.assess_utility(potential_contact=neighbor)                    
                        if neighbor_utility > utility_threshold: # Potential contact assess agent utility
                            self_utility = self.model.assess_utility(potential_contact=self)
                            if self_utility > utility_threshold:
                                # Add the neighbor to the agent's social contacts, and vice versa
                                self.social_contacts.append(neighbor)
                                neighbor.social_contacts.append(self)
                        
                        self.interaction_complete = True
                        break  # Ensure only one interaction happens per step

        # Define function for contact activation to initiate resource request
        def decide_contact_activation(self):
            activation_probability = 0.5 if self.LA == 1 else 0.25
            if self.GRE > context and self.gender == 0:
                activation_probability += 0.1  # Men with high GRE are more likely to activate 
            if self.gender == 1:
                activation_probability += 0.2  # Women are more likely to activate a contact (Brashears et al., 2016)
            self.contact_activated = random.random() < activation_probability

            # If contact activated, begin resource request legitimacy assessment
            if self.contact_activated:
                if self.social_contacts:
                    selected_contact = random.choice(self.social_contacts)
                    self.assess_legitimacy_request(selected_contact)  
            self.request_complete = True        

        # Define function for legitimacy assessment, legitimacy is function of RCB, SC, gender, and request match
        def assess_legitimacy_request(self, selected_contact):
            role_consistent_behavior = self.RCB
            sc_factor = self.SC 
            if self.gender == selected_contact.gender:
                nature_of_request = 1
            else:
                nature_of_request = 0.5
            legitimacy = (role_consistent_behavior * 0.4) + (sc_factor * 0.4) + (nature_of_request * 0.2)
            if self.gender == 1 and context >= 0.65:
                    legitimacy -= 0.2
            if legitimacy > utility_threshold:
                self.evaluate_resource_request(selected_contact)

        # Define function for the contact to assess the resource request's
        def evaluate_resource_request(self, selected_contact):
            if selected_contact.gender == 0 and selected_contact.resources <= 0.12:
                response_legitimacy = 0
            elif selected_contact.gender == 1 and selected_contact.resources <= 0.12:
                response_legitimacy = 0
            else: # Incorporates variables from both the agent and contact
                role_consistent_behavior_requester = self.RCB
                sc_factor_requester = self.SC
                role_consistent_behavior_contact = selected_contact.RCB
                sc_factor_contact = selected_contact.SC
                nature_of_request = 1 if self.gender == selected_contact.gender else 0.5
            
            # Adjust nature-of-request based on selected contact's RCB and gender mismatch
                if role_consistent_behavior_contact > 0.7 and self.gender != selected_contact.gender:
                    nature_of_request *= 0.8  
                response_legitimacy = (
                    (role_consistent_behavior_requester * 0.3) + 
                    (sc_factor_requester * 0.3) + 
                    (nature_of_request * 0.1) + 
                    (role_consistent_behavior_contact * 0.3)
                )
                # Apply gender-based biases for resource allocation
                if self.gender == 1 and random.random() < 0.3:  
                    if context >= 0.65:
                        response_legitimacy -= 0.2
           
            # Check if response legitimacy exceeds the utility threshold
            if response_legitimacy > utility_threshold:
                # Transfer resources
                transfer_amount = 0.01  
                selected_contact.resources -= transfer_amount
                self.resources += transfer_amount
                self.RS += 1  # Mark resources as shared
                selected_contact.RG += 1 # mark resources as given
            else:
                # If resources were not shared, consider severing the relationship
                if len(selected_contact.social_contacts) > 2:  # Ensure agent doesn’t drop below 2 contacts
                    if random.random() < 0.5:  # 50% chance to sever the relationship
                        selected_contact.social_contacts.remove(self)
                        self.social_contacts.remove(selected_contact)
                        selected_contact.relationship_assess = True
                        self.relationship_assess = True

        # Define function to decide if the agent wants to maintain relationships
        def assess_relationships(self):
            if not self.relationship_assess:
                # Only assess relationships if agent has more than 2 social contacts
                if len(self.social_contacts) > 2:
                    # Randomly pick one contact to assess based on resources given to shared
                    contact = random.choice(self.social_contacts)
                    if self.RS < contact.RG:
                        sever_chance = 0.5
                    else:
                        sever_chance = 0.25
                    # Apply severing logic
                    if random.random() < sever_chance:
                        self.social_contacts.remove(contact)
                        contact.social_contacts.remove(self)
                # Mark the relationship assessment as complete
                self.relationship_assess = True

        # Define function to update agents' locations based on org level
        def move_to_level_location(self):
            region_width = self.model.grid.width // 8
            min_x = (self.level - 1) * region_width
            max_x = self.level * region_width
            new_x = random.randint(min_x, max_x - 1)
            new_y = random.randint(0, self.model.grid.height - 1)
            self.model.grid.move_agent(self, (new_x, new_y))

        # Define function to update social capital based on new network and RCB
        def update_social_capital(self):
            self.previous_SC = self.SC  
            level_factor = self.level / 8
            kin_factor = 1 - (self.KC / max_connections)
            if context >= 0.65 and self.gender == 1: 
                kin_penalty = context * 0.2
                kin_factor *= (1 - kin_penalty)
            elif context < 0.65 and self.gender == 0:
                kin_boost = (1 - context) * 0.1
                kin_factor *= (1 + kin_boost)
            rcb_factor = self.RCB
            # Women penalized for low RCB in male dominated spaces
            if context >= 0.65:
                if self.gender == 1: 
                    rcb_factor *= (1 - context * 0.2)
                else:
                    rcb_factor *= (1 + context * 0.1)  

            if len(self.social_contacts) > 0:
                network_factor = len(self.social_contacts) / self.max_con  
            else:
                network_factor = 0
            # In male-dominated contexts, all factors are weighted equally. In egalitarian/female-dominated contexts, level and network are weighted more heavily
            if context >= 0.65:
                self.SC = (level_factor + kin_factor + network_factor + rcb_factor) / 4
            else:
                self.SC = (level_factor * 0.3 + network_factor * 0.3 + kin_factor * 0.2 + rcb_factor * 0.2)

            self.SC = min(1, max(0, self.SC))

            total_SC = sum([agent.SC for agent in self.model.schedule.agents])
            if total_SC > max_SC:
                scaling_factor = max_SC / total_SC
                self.SC *= scaling_factor

        # Define order of agent operations
        def step(self):
            self.calculate_RCB()
            self.interaction_complete = False
            self.request_complete = False
            self.relationship_assess = False

            self.interact_with_nearby_agents()
            if not self.request_complete:
                self.decide_contact_activation()

            self.assess_relationships()
            self.update_social_capital()

    # Define NEX-GL model steps
    class CustomModel(Model):
        # Initialize rules of model and their sequence
        def __init__(self):
            super().__init__()
            self.schedule = StagedActivation(self, stage_list=[
                "calculate_RCB",
                "interact_with_nearby_agents",
                "decide_contact_activation",
                "assess_relationships",
                "update_social_capital"
            ])
            self.grid = MultiGrid(world_width, world_height, True)  # Toroidal grid (wraps around)
            self.industry_type = determine_industry_type(context)
            self.step_count = 0
            self.init_agents()
            self.assign_initial_contacts()

        # Number of iterations (steps) the model has run, incremented each iteration
        def step(self):
            self.schedule.step()  # Move the model forward one step
            self.step_count += 1
        
        # Gather all needed data from the model
        def collect_data(self, step_count):
            for agent in self.schedule.agents:
                data['step'].append(step_count)
                data['agent_id'].append(agent.unique_id)
                data['gender'].append(agent.gender)
                data['level'].append(agent.level)
                data['SC'].append(round(agent.SC, 3))
                data['resources'].append(round(agent.resources, 3))
                data['RG'].append(agent.RG)
                data['RCB'].append(round(agent.RCB,3))
                data['KC'].append(agent.KC)
                data['NO'].append(round(agent.NO, 3))
                data['PM'].append(round(agent.PM, 3))
                data['PW'].append(round(agent.PW, 3))
                data['DC'].append(agent.DC)

        # Initialize agents in the simulation
        def init_agents(self):
            agent_id = 0
            for level, count in level_counts.items():
                for _ in range(count):
                    agent = CustomAgent(agent_id, self, level)
                    x = self.random.randrange(self.grid.width)
                    y = self.random.randrange(self.grid.height)
                    agent.setup_agent()
                    self.grid.place_agent(agent, (x, y))
                    agent.move_to_level_location()
                    self.schedule.add(agent)
                    agent_id += 1
            for agent in self.schedule.agents:
                agent.move_to_level_location()
        
        # Call previously defined function to assign initial network contacts
        def assign_initial_contacts(self):
            for agent in self.schedule.agents:
                agent.initial_contacts()
                
        # Define function to allow agents to assess the utility of potential contacts and vice versa
        def assess_utility(self, potential_contact):
            sc_factor = potential_contact.SC
            level_factor = potential_contact.level / 8
            # Gender factor: Men have more advantage in male-dominated contexts
            if potential_contact.gender == 0:  # Male
                gender_factor = (1 + context) / 2
            else:  # Female
                gender_factor = 1 - (context / 2)
            # Kin contacts factor: Women are penalized more in male-dominated contexts
            if potential_contact.gender == 1:  # Female
                kc_factor = 1 - (min(1, potential_contact.KC / max_connections) * context)
            else:  # Male
                kc_factor = 1  # No impact for men
            utility = (sc_factor * 0.4) + (level_factor * 0.3) + (gender_factor * 0.15) + (kc_factor * 0.15)
            return min(max(utility, 0), 1)

        # Define function to promote or demote agents based on the change in the agent's social capital across runs
        def promote_or_demote(self, agent):
            SC_diff = agent.SC - agent.previous_SC  
            if SC_diff > 0 and agent.level < 8: # Promotion logic
                if random.random() < SC_diff*10:  
                    agent.level += 1
                    agent.move_to_level_location()
            elif SC_diff < 0 and agent.level > 1:  # Demotion logic
                if random.random() < (-SC_diff)*10:
                    agent.level -= 1
                    agent.move_to_level_location()

        # Define function to balance the levels and maintain a heirarchical structure
        def balance_levels(self):
            total_levels = 8  
        # Iterate from the highest level down to the lowest level
            for current_level in range(total_levels, 0, -1):
                current_count = sum(1 for agent in self.schedule.agents if agent.level == current_level)
                target_count = level_counts[current_level]
            # If there are too many agents at the current level, demote the ones with the lowest SC
                if current_count > target_count:
                    surplus = current_count - target_count
                    candidates = sorted([agent for agent in self.schedule.agents if agent.level == current_level and agent.SC < np.percentile([a.SC for a in self.schedule.agents if a.level == current_level], 75)],
                                    key=lambda agent: agent.SC)
                    for _ in range(surplus):
                        if candidates:
                        # Select the candidate with the lowest SC for demotion
                            candidate = candidates.pop(0)
                            candidate.level -= 1
                            candidate.move_to_level_location()

            # If there are too few agents at the current level, promote agents from the next lower level
                elif current_count < target_count and current_level > 1:
                    deficit = target_count - current_count

                # Get candidates for promotion (highest SC agents from the next lower level)
                    candidates = sorted([agent for agent in self.schedule.agents if agent.level == current_level - 1],
                                    key=lambda agent: agent.SC, reverse=True)
                    for _ in range(deficit):
                        if candidates:
                        # Select the candidate with the highest SC for promotion
                            candidate = candidates.pop(0)
                            candidate.level += 1
                            candidate.move_to_level_location()

        # Define function to calculate network openness for data collection
        # The network effective size based on Burt’s (1992) definition, calculated using Borgatti’s (1997) expression and is the ego’s degree minus average degree of alters
        def calculate_network_openness(self, agent):
            agent.DC =len(agent.social_contacts)
            ego_degree = len(agent.social_contacts)
            if ego_degree == 0:
                agent.NO = 0
            else:
                total_alter_degree = sum(len(contact.social_contacts) for contact in agent.social_contacts)
                avg_alter_degree = total_alter_degree / ego_degree
                agent.NO = ego_degree - avg_alter_degree

        # Define function to calculate the proportion of female/male contacts for data collection
        def update_contact_proportions(self, agent):
            male_contacts = sum(1 for contact in agent.social_contacts if contact.gender == 0)
            total_contacts = len(agent.social_contacts)
            if total_contacts > 0:
                agent.PM = male_contacts / total_contacts
                agent.PW = 1 - agent.PM
            else:
                agent.PM = 0
                agent.PW = 0

        # Define function with the order of operations for each agent behavior 
        def step(self):
            # Each agent takes their step in sequence
            self.schedule.step()  
            # After all agents take a step, balance the levels
            for agent in self.schedule.agents:
                self.promote_or_demote(agent)
                self.calculate_network_openness(agent)
                self.update_contact_proportions(agent)
            self.balance_levels()
            self.step_count += 1
          

# Run the model with each parameter combination and collect data every 60 (n) steps
    model = CustomModel()
    for i in range(max_steps):  
        print(f"Running step {i+1}")
        model.step()
        n = 60
        if i % n == 0:
                model.collect_data(i) 

    df = pd.DataFrame(data)
    df['context'] = context
    df['mean_kin_m'] = mean_kin_m
    df['mean_kin_w'] = mean_kin_w
    df['contact_mean'] = contact_mean
    return df


# Set possible parameter values
context_values = [0.1, 0.35, 0.5, .65, 0.8, 0.9]  # Values for context parameter
mean_kin_m_values = [19, 22, 25]  # Values for mean kin contacts for men
mean_kin_w_values = [32, 36, 40]  # Values for mean kin contacts for women
sd_kin_m_values = [3]  # Standard deviation of kin contacts for men
sd_kin_w_values = [5]  # Standard deviation of kin contacts for women
contact_mean_values = [9, 12, 15]  # Average number of social contacts
contact_sd_values = [3]  # Standard deviation of social contacts
max_connections_values = [100]  # Max connections in the network taking from Dunbar's number

# Generate all possible combinations of the parameters for simulation runs
parameter_combinations = list(itertools.product(
    context_values,
    mean_kin_m_values,
    mean_kin_w_values,
    sd_kin_m_values,
    sd_kin_w_values,
    contact_mean_values,
    contact_sd_values,
    max_connections_values
))

# Prepare the unique parameter sets for simulation
parameter_sets = []
for combination in parameter_combinations:
    parameter_set = {
        'context': combination[0],
        'mean_kin_m': combination[1],
        'mean_kin_w': combination[2],
        'sd_kin_m': combination[3],
        'sd_kin_w': combination[4],
        'contact_mean': combination[5],
        'contact_sd': combination[6],
        'max_connections': combination[7]
    }
    parameter_sets.append(parameter_set)

# Create function to run simulations with different parameter sets
def run_multiple_simulations(parameter_sets, max_steps=300):
    all_data = []
    for i, params in enumerate(parameter_sets):
        df = run_simulation(**params, max_steps=max_steps)
        df['simulation_run'] = i + 1  # Tag each simulation run for later identification
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Run multiple simulations
combined_results = run_multiple_simulations(parameter_sets, max_steps=300)

# Save combined results to CSV
combined_results.to_csv('combined_simulation_results.csv', index=False)

# Visually check results the results
print(combined_results.head())
