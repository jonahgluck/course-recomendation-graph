from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx

app = Flask(__name__)

courses = {
    'ENG EK 125': {'prerequisites': [], 'skills': ['Mathematical Skills', 'Problem Solving']},
    'ENG EK 210': {'prerequisites': ['ENG EK 125'], 'skills': ['Engineering Fundamentals', 'Technical Writing']},
    'ENG ME 231': {'prerequisites': ['ENG EK 210'], 'skills': ['Mechanical Engineering Skills', 'Design Skills']},
    'ENG EC 327': {'prerequisites': ['ENG EK 125'], 'skills': ['Embedded Systems', 'Programming Skills']},
    'ENG EC 464': {'prerequisites': ['ENG EC 327'], 'skills': ['Digital Signal Processing', 'Advanced Programming']},
    'ENG BE 403': {'prerequisites': ['ENG EK 125'], 'skills': ['Biomedical Engineering Principles', 'Analytical Skills']},
    'ENG SE 501': {'prerequisites': ['ENG EK 210'], 'skills': ['Systems Engineering Concepts', 'Modeling Skills']},

    'CAS CS 111': {'prerequisites': [], 'skills': ['Mathematical Skills', 'Introductory Programming', 'Algorithmic Thinking']},
    'CAS CS 112': {'prerequisites': ['CAS CS 111'], 'skills': ['Data Structures', 'Algorithmic Thinking']},
    'CAS CS 210': {'prerequisites': ['CAS CS 112'], 'skills': ['Computer Systems', 'Programming Skills']},
    'CAS CS 330': {'prerequisites': ['CAS CS 210'], 'skills': ['Theory of Computation', 'Problem Solving']},
    'CAS CS 455': {'prerequisites': ['CAS CS 330'], 'skills': ['Machine Learning', 'Data Analysis']},
    'CAS CS 460': {'prerequisites': ['CAS CS 210'], 'skills': ['Data Visualization', 'Information Design']},
    'CAS CS 411': {'prerequisites': ['CAS CS 210'], 'skills': ['Software Engineering', 'Project Management']},
    'CAS CS 440': {'prerequisites': ['CAS CS 210'], 'skills': ['Artificial Intelligence', 'Problem Solving']},

    'CAS EC 101': {'prerequisites': [], 'skills': ['Economic Theory', 'Analytical Skills']},
    'CAS EC 102': {'prerequisites': [], 'skills': ['Economic Theory', 'Critical Thinking']},
    'CAS EC 201': {'prerequisites': ['CAS EC 101', 'CAS MA 123'], 'skills': ['Microeconomics', 'Quantitative Analysis']},
    'CAS EC 202': {'prerequisites': ['CAS EC 102', 'CAS MA 124'], 'skills': ['Macroeconomics', 'Analytical Skills']},
    'CAS EC 320': {'prerequisites': ['CAS EC 201'], 'skills': ['Econometrics', 'Data Analysis']},
    'CAS EC 414': {'prerequisites': ['CAS EC 320'], 'skills': ['Behavioral Economics', 'Policy Analysis']},

    'CAS PS 101': {'prerequisites': [], 'skills': ['Psychological Concepts', 'Critical Thinking']},
    'CAS PS 231': {'prerequisites': ['CAS PS 101'], 'skills': ['Developmental Psychology', 'Research Skills']},
    'CAS PS 241': {'prerequisites': ['CAS PS 101'], 'skills': ['Social Psychology', 'Interpersonal Skills']},
    'CAS PS 371': {'prerequisites': ['CAS PS 241'], 'skills': ['Cognitive Psychology', 'Analytical Skills']},
    'CAS PS 332': {'prerequisites': ['CAS PS 231'], 'skills': ['Neuroscience', 'Analytical Thinking']},

    'CAS PH 100': {'prerequisites': [], 'skills': ['Philosophical Thinking', 'Ethical Reasoning']},
    'CAS PH 110': {'prerequisites': [], 'skills': ['Logic', 'Analytical Thinking']},
    'CAS PH 150': {'prerequisites': [], 'skills': ['Ethics', 'Moral Reasoning']},
    'CAS PH 245': {'prerequisites': ['CAS PH 100'], 'skills': ['Philosophy of Mind', 'Critical Analysis']},
    'CAS PH 260': {'prerequisites': ['CAS PH 110'], 'skills': ['Epistemology', 'Logical Reasoning']},
    'CAS PH 300': {'prerequisites': ['CAS PH 150'], 'skills': ['Metaphysics', 'Abstract Thinking']}
}

skill_categories_to_jobs = {
    'Mathematical Skills': ['Data Analyst', 'Engineer', 'Actuary'],
    'Problem Solving': ['Engineer', 'Consultant', 'Data Scientist'],
    'Engineering Fundamentals': ['Mechanical Engineer', 'Project Manager'],
    'Technical Writing': ['Technical Writer', 'Project Manager'],
    'Mechanical Engineering Skills': ['Mechanical Engineer', 'Robotics Engineer'],
    'Design Skills': ['Product Designer', 'Mechanical Engineer'],
    'Embedded Systems': ['Embedded Software Engineer', 'Robotics Engineer'],
    'Digital Signal Processing': ['Signal Processing Engineer', 'Software Developer'],
    'Biomedical Engineering Principles': ['Biomedical Engineer', 'Clinical Engineer'],
    'Systems Engineering Concepts': ['Systems Engineer', 'Operations Analyst'],

    'Introductory Programming': ['Software Developer', 'IT Support Specialist'],
    'Algorithmic Thinking': ['Software Engineer', 'Data Scientist'],
    'Data Structures': ['Software Engineer', 'Systems Analyst'],
    'Computer Systems': ['Systems Engineer', 'Software Developer'],
    'Theory of Computation': ['Software Engineer', 'Computational Scientist'],
    'Machine Learning': ['Machine Learning Engineer', 'Data Scientist'],
    'Data Analysis': ['Data Analyst', 'Business Intelligence Analyst'],
    'Data Visualization': ['Data Visualization Specialist', 'UI/UX Designer'],
    'Programming Skills': ['Software Developer', 'Data Scientist'],
    'Software Engineering': ['Software Engineer', 'DevOps Engineer'],
    'Project Management': ['Project Manager', 'Product Manager'],
    'Artificial Intelligence': ['AI Researcher', 'Machine Learning Engineer'],

    'Economic Theory': ['Economic Consultant', 'Policy Analyst'],
    'Critical Thinking': ['Policy Analyst', 'Researcher'],
    'Microeconomics': ['Financial Analyst', 'Economic Consultant'],
    'Macroeconomics': ['Policy Analyst', 'Economic Consultant'],
    'Quantitative Analysis': ['Data Analyst', 'Financial Analyst'],
    'Econometrics': ['Econometrician', 'Data Scientist'],
    'Policy Analysis': ['Policy Analyst', 'Economic Researcher'],
    'Behavioral Economics': ['Behavioral Economist', 'Market Research Analyst'],
}

G = nx.DiGraph()
for course in courses:
    G.add_node(course, layer=0)
for skill_category in skill_categories_to_jobs:
    G.add_node(skill_category, layer=1)
for jobs in skill_categories_to_jobs.values():
    for job in jobs:
        G.add_node(job, layer=2)
for course, details in courses.items():
    for prereq in details['prerequisites']:
        if prereq in G:
            G.add_edge(prereq, course, relationship='prerequisite')
    for skill in details['skills']:
        if skill in G:
            G.add_edge(course, skill, relationship='course_to_skill')
for skill_category, jobs in skill_categories_to_jobs.items():
    for job in jobs:
        if job in G:
            G.add_edge(skill_category, job, relationship='skill_to_job')

node_mapping = {node: idx for idx, node in enumerate(G.nodes)}
edges = [(node_mapping[u], node_mapping[v]) for u, v in G.edges]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
node_features = torch.eye(len(G.nodes))
data = Data(x=node_features, edge_index=edge_index)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN(input_dim=len(G.nodes), hidden_dim=16, output_dim=len(G.nodes))
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

def suggest_next_steps(current_courses=[]):
    next_courses = set()
    next_skills = set()
    next_jobs = set()

    current_skills = set()
    for course in current_courses:
        if course in courses:
            current_skills.update(courses[course]['skills'])

    for course in current_courses:
        if course in G:
            for neighbor in G.neighbors(course):
                if G.nodes[neighbor]['layer'] == 1:
                    next_skills.add(neighbor)
                elif G.nodes[neighbor]['layer'] == 0:
                    next_courses.add(neighbor)

    for skill in current_skills:
        if skill in G:
            for neighbor in G.neighbors(skill):
                if G.nodes[neighbor]['layer'] == 2:
                    next_jobs.add(neighbor)

    return {
        'next_courses': list(next_courses),
        'next_skills': list(next_skills),
        'next_jobs': list(next_jobs)
    }

@app.route('/')
def index():
    all_jobs = set()
    for jobs in skill_categories_to_jobs.values():
        all_jobs.update(jobs)

    all_jobs = sorted(all_jobs)

    return render_template('index.html', courses=courses.keys(), jobs=all_jobs)


@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    current_courses = request.form.getlist('courses')
    selected_jobs = request.form.getlist('jobs')
    suggestions = suggest_next_steps(current_courses)
    return {
        'next_courses': suggestions['next_courses'],
        'next_skills': suggestions['next_skills'],
        'next_jobs': suggestions['next_jobs']
    }

if __name__ == '__main__':
    app.run(debug=True)
