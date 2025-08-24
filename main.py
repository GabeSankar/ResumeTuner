import numpy as np
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
import re
import heapq
class VectorSpaceMath():

    def __init__(self):
        #initialize sentance transformer
        self.sentance_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def gaussian_kernel(self, x, y, sigma=1.0):
        """Compute Gaussian (RBF) kernel between x and y."""
        pairwise_sq_dists = cdist(x, y, 'sqeuclidean')
        return np.exp(-pairwise_sq_dists / (2 * sigma ** 2))

    def compute_weighted_maxmimum_mean_discrepancy(self, X, Y, sigma, w_X = None, w_Y = None):
        """Running Maximum Mean Discrepancy between embeddings with weights to emphasize job title
        over tasks within job to analyze most important similarity
        
        inputs are:
        X, which is the array of n samples by the feature vectors
        Y, which is the array of n samples by the feature vectors of the other distribution
        w_X, which is the optional weights for X
        w_Y, which is the optional weights for Y
        weights is 1d numpy tensor
        sigma, which is the bandwith of the gaussian kernel, controls standard deviation"""
        K_XX = self.gaussian_kernel(X, X, sigma)
        K_YY = self.gaussian_kernel(Y, Y, sigma)
        K_XY = self.gaussian_kernel(X, Y, sigma)

        #Normalize the weights
        n_x, n_y = X.shape[0], Y.shape[0]
        #initialize equal weights if no weights
        if w_X is None:
            w_X = np.ones(n_x) / n_x
        else:
            w_X = w_X / np.sum(w_X)
        
        if w_Y is None:
            w_Y = np.ones(n_y) / n_y
        else:
            w_Y = w_Y / np.sum(w_Y)

        #Compute squared maximum mean discrepancy with weighted sums
        mmd = np.sum(K_XX * np.outer(w_X, w_X)) + np.sum(K_YY * np.outer(w_Y, w_Y)) - 2 * np.sum(K_XY * np.outer(w_X, w_Y))
        #Compute maximum mean discrepancy
        return np.sqrt(mmd)
    
    def cosine_similarity(self, a, b):
        """Compute the cosine similarity between two embedding"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0  #No division by zero
        return np.dot(a, b) / (a_norm * b_norm)
    
    def average_embedding(self, embeddings):
        """Averages embeddings along first dimension with input of shape (n_samples, embedding dim)"""
        return np.mean(embeddings, axis=0)
    
    def create_embeding_vectors(self, texts):
        """Creates embeddings with sentance transformer for list of texts"""
        return self.sentance_transformer.encode(texts)


class LatexAndJsonHandler():
    """Class to handle generation of latex as well as processing resume data"""
    def __init__(self):
        self.vector_handler = VectorSpaceMath()

    def generate_latex_experience(experience_data):
        """from experience json structure it creates single experience bullet"""
        experience = f"\\textbf{{{experience_data["company"]}}}, {experience_data["location"]} --- \\textit{{{experience_data["title"]}}} \\hfill {experience_data["dates"]}\n\\begin{{itemize}}\n"
        
        for responsibility in experience_data["responsibilities"]:
            experience += f"\\item {responsibility}\n"

        experience += "\\end{itemize}"
        
        return experience

    def generate_latex_education(education_data, skills_data, awards_data):
        """from experience json structure, it creates the education, skills and awards header"""
        education_header = f"\\textbf{{{education_data["school"]}}}, {education_data["location"]} \\hfill {education_data["dates"]}\\\\\n{education_data["degree"]}\\\\\nConcentrations: {', '.join(map(str, education_data["concentrations"]))}\\\\\nCurrent Coursework: {', '.join(map(str, education_data["current_coursework"]))}\\\\\nRelevant Coursework: {', '.join(map(str, education_data["relevant_coursework"]))}\\\\\nGPA: {education_data["gpa"]} \\\\\n\n\\vspace{{0.3em}}\n" 

        skills = f"\\textbf{{Skills:}} {', '.join(map(str, skills_data))}\n"
        awards = f"\\textbf{{Awards:}} {', '.join(map(str, awards_data))}\n"
        
        return education_header + skills + awards

    def generate_latex_projects(project_data):
        """from experience json structure it creates single project bullet"""
        project = f"\\textbf{{{project_data["name"]}}} --- {project_data["description"]} \\hfill {project_data["dates"]}\n\\begin{{itemize}}\n"
        
        for detail in project_data["details"]:
            project += f"\\item {detail}\n"

        project += "\\end{itemize}"
        
        return project


    def generate_latex_leadership(leadership_data):
        """from experience json structure it creates single leadership bullet"""
        leadership = f"\\textbf{{{leadership_data["organization"]}}}, {leadership_data["location"]} --- {leadership_data["title"]} \\hfill {leadership_data["dates"]}\n\\begin{{itemize}}\n"
        
        for achievement in leadership_data["achivements"]:
            leadership += f"\\item {achievement}\n"

        leadership += "\\end{itemize}"
        
        return leadership

    def generate_latex_additional_skills(languages_data, additional_skills_data):
        skills = f"Additional Skills: {', '.join(map(str, additional_skills_data))} \\\\\n"
        languages = f"Languages: {', '.join(map(str, languages_data))}\n"

        return languages + skills

    def load_data_and_vectors(self, json):

        exp_clusters = {}
        exp_vectorized_clusters = {}
        for experience in json["experience"]:
            # experience["title"] + experience["dates"] + experience["company"] is the format to key the clusters in case of return to a position or the same positions
            experience_key = experience["title"] + experience["dates"] + experience["company"]
            exp_clusters[experience_key] = experience["responsibilities"]
            exp_vectorized_clusters[experience_key] = self.vector_handler.create_embeding_vectors(experience["responsibilities"])

        proj_clusters = {}
        proj_vectorized_clusters = {}
        for project in json["projects"]:
            project_key = project["name"] + project["dates"] + project["description"]
            proj_clusters[project_key] = project["responsibilities"]
            proj_vectorized_clusters[project_key] = self.vector_handler.create_embeding_vectors(project["responsibilities"])


        leadership_clusters = {}
        leadership_vectorized_clusters = {}
        for leadership in json["leadership_and_community_engagement"]:
            leadership_key = leadership["title"] + leadership["dates"] + leadership["organization"]
            leadership_clusters[leadership_key] = leadership["responsibilities"]
            leadership_vectorized_clusters[leadership_key] = self.vector_handler.create_embeding_vectors(leadership["responsibilities"])

        #loading lists for skills, languages, and awards
        skills = json["skills"]["technical"]
        vectorized_skills = self.vector_handler.create_embeding_vectors(skills)
        skills_dict = dict(zip(skills, vectorized_skills))

        languages = json["skills"]["languages"]
        vectorized_languages = self.vector_handler.create_embeding_vectors(languages)
        languages_dict = dict(zip(languages, vectorized_languages))

        add_skills = json["skills"]["additional_skills"]
        vectorized_add_skills = self.vector_handler.create_embeding_vectors(add_skills)
        add_skills_dict = dict(zip(add_skills, vectorized_add_skills))

        awards = json["awards"]
        vectorized_awards = self.vector_handler.create_embeding_vectors(awards)
        awards_dict = dict(zip(awards, vectorized_awards))

        #load eductation
        
    def rank_clusters(self, vectorized_clusters, job_description_vectors, n_ranked):
        #uses MMD so smallest shows least discrepancy
        mmd_clusters = {}
        for key in vectorized_clusters:
            #vstack to convert list to vectors
            mmd = self.vector_handler.compute_weighted_maxmimum_mean_discrepancy(np.vstack(vectorized_clusters[key]), np.vstack(job_description_vectors))
            mmd_clusters[mmd] = {key : vectorized_clusters[key]}
        
        smallest = heapq.nsmallest(n_ranked, mmd_clusters.items(), key=lambda x: x[0])
        return smallest

