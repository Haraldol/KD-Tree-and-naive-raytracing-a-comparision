"""
KD-Tree raytracer benchmarking program

This program compares the running time of a naive raytracer to my own ray tracer algorithm using a KDTree from the scipy.spatial library.
The naive ray tracer is based on Omar Aflaks ray tracer,
available on Medium: https://omaraflak.medium.com/ray-tracing-from-scratch-in-python-41670e6a96f9

Author: Harald Shiva Olin
Date: 2025-01-10
"""
import matplotlib
matplotlib.use('TkAgg') # Added for showing plots using the tk-interface 
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import time
import os

def normalize_vector(vector):
    """
    This function takes a vector as an argument and normalizes it: vector=vector/|vector|
        Args:
            vector: 3D vector
        Returns:
            a normalized vector       
    """ 
    return vector / np.linalg.norm(vector)

def sphere_ray_intersection(center, radius, ray_origin, ray_direction):
    """
    Detects the intersection between sphere and ray by solving:
    |ray(t) - C|^2 = r ^2 for t, => t^2+bt+c=0, where ray(t)= ray_origin + ray_direction*t
    and analyzing the discriminant: delta = b^2 - 4*c
        Args:
            center: The center of the sphere
            radius: The radius of the sphere
            ray_origin: The origin of the ray
            ray_direction:The direction of the ray
        Returns: 
            returns the scaling factor t in ray(t) that determines where the closest intersection with the sphere occurs
    """
    b = 2 * np.dot(ray_direction, ray_origin - center) 
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2 
    delta = b **2 - 4 * c 
    if delta > 0: 
        t1 = (-b +np.sqrt(delta))/2
        t2 = (-b - np.sqrt(delta))/2
        if t1 > 0 and t2 > 0:
            return min(t1,t2) 
    return None

def reflected_ray_direction(vector, axis):
    """
    Calculates the direction of the reflected rays direction 
        Args:
            vector (vec3): the incoming director of the ray to be reflected
            axis (vec3): the normal to the surface of the ray stroke
        Return:
            (vec3) the reflected ray 
        
    """
    return vector - 2 * np.dot(vector, axis) * axis

def closest_intersection_naive(spheres, ray_origin, ray_direction):
    """
    Searches all spheres in the scene for intersection, and finds the sphere with the closest distance to the camera
        Args:
            spheres: a list of all the spheres in the scene
            ray_origin: vec3 origin of the ray
            ray_direction: vec3 direction of the ray
        Returns: 
            closest_sphere: sphere object that is the closest to the camera
            min_distance: float the distance to the closest sphere
            
    """
    distances = [sphere_ray_intersection(sphere['center'], sphere['radius'], ray_origin, ray_direction) for sphere in spheres]
    closest_sphere = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            closest_sphere = spheres[index]
    return closest_sphere, min_distance



def closest_intersection_kd_tree(spheres, ray_origin, ray_direction, kd_tree,nearest_point):
    """
    Searches all spheres in order of closest to the nearest point, and assumes that the sphere nearest to the "nearest_point" is the closest intersection
        Args: 
            spheres: a list of all the spheres in the scene 
            ray_orogin: vec3 origin of the ray
            ray_direction: vec3 direction of the ray 
        Returns: 
            closest_sphere: sphere object that is the closest to the camera
            min_distance: float the distance to the closest sphere
    """
    distances, indices = kd_tree.query(nearest_point, k=len(spheres), distance_upper_bound=np.inf) # distance_upper_bound set to np.inf to render all spheres

    closest_sphere = None
    min_distance = np.inf
    for i in indices:  
        distance = sphere_ray_intersection(spheres[i]['center'], spheres[i]['radius'], ray_origin, ray_direction)
        if distance is not None and distance < min_distance:
            min_distance = distance
            closest_sphere = spheres[i]
            return closest_sphere,min_distance

    return closest_sphere, min_distance


def create_evenly_spaced_spheres(num_spheres_x, num_spheres_y, radius, screen):
    """
    Automates the process of filling the test scene with spheres, creates evenly spaced spheres.
        Args:
            num_spheres_x: number of spheres in x direction
            num_spheres_y: number of spheres in y direction
            radius: radius of the spheres
            screen: the render space 
        Returns:
            the list of evenly spaced spheres
    """
    left, top, right, bottom = screen
    spheres = []

    # Calculate the spacing between sphere centers
    spacing_x = (right - left) / (num_spheres_x - 1) if num_spheres_x > 1 else 0
    spacing_y = (bottom - top) / (num_spheres_y - 1) if num_spheres_y > 1 else 0

    for i in range(num_spheres_y):
        for j in range(num_spheres_x):
            x = left + j * spacing_x
            y = top + i * spacing_y
            z = -0.2 - j * spacing_x # initiate spheres on the z=-0.2 plane

            # Calculate ambient color based on world position
            ambient = np.array([abs(x)/2, abs(y)/2, 0.1])
            diffuse = np.where(ambient!=0, 0.7,0)
            center = np.array([x, y, z])

            spheres.append({
                'center': center,
                'radius': radius,
                'ambient': ambient,
                'Diffuse': diffuse,
                'specular': np.array([1, 1, 1]),
                'shininess': 100,
                'reflection': 0.5
            })
    # append the sphere that represents the "floor"
    spheres.append({ 'center': np.array([0, -9000, 0]),
                     'radius': 9000 - 0.7, 
                     'ambient': np.array([0.1, 0.1, 0.1]),
                    'Diffuse': np.array([0.6, 0.6, 0.6]),
                    'specular': np.array([1, 1, 1]),
                    'shininess': 100, 'reflection': 0.5})
    return spheres


def render(screen,width,height,spheres,kd_tree,model,image):
    """
    Renders the program with the decired model, kdtree or naive.
    """
    #init
    camera = np.array([0, 0, 1])
    max_depth = 3 # number of bounces
    light = {'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'Diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1])}
    


    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):  # divides the the top and bottom edges into height number of evenly spaced points
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)): # same for x coordinate
            pixel = np.array([x, y, 0]) # a pixel on the screen, the screen is set where z=0
            ray_origin = camera
            ray_direction = normalize_vector(pixel - ray_origin) # normalizes the vector from ray_origin to current pixel

            color = np.zeros((3))
            reflection = 1  #reflection start at 1 for maximum light, and decreases for each depth == "bounce of the ray"

            for k in range(max_depth):

                #looking for the closest intersection using kd_tree or naive raytracing
                if( model == "kdtree"):
                    if(k==0):
                        closest_sphere, min_distance = closest_intersection_kd_tree(spheres, ray_origin, ray_direction,kd_tree,pixel)
                    else:
                        closest_sphere, min_distance = closest_intersection_kd_tree(spheres, ray_origin, ray_direction,kd_tree,ray_origin)
                    if closest_sphere is None or min_distance is None:
                        break
                else:
                    closest_sphere, min_distance = closest_intersection_naive(spheres, ray_origin, ray_direction)
                    if closest_sphere is None or min_distance is None:
                        break

                #intersectionpoint between ray and closest sphere
                intersection = ray_origin +  ray_direction * min_distance

                normalized_to_surface = normalize_vector(intersection - closest_sphere['center'])
                shifted_point = intersection + 1e-5 * normalized_to_surface #move a tiny step up the normal to avoid detecting the same intersection point again
                lights_intersection = normalize_vector(light['position'] - shifted_point)

                # calculate the light inter
                if(model == "kdtree"):
                    _, min_distance = closest_intersection_kd_tree(spheres, shifted_point, lights_intersection,kd_tree,shifted_point)
                else:
                    _, min_distance = closest_intersection_naive(spheres, shifted_point, lights_intersection)
                
                lights_intersection_distance = np.linalg.norm(light['position']-intersection)
                shadowed = min_distance < lights_intersection_distance
                if shadowed:
                    break

                #RGB and blinn-Phong model
                illumination = np.zeros((3))
                #ambient
                illumination += closest_sphere['ambient']*light['ambient']
                #diffuse
                illumination += closest_sphere['Diffuse'] * light['Diffuse'] * max(0, np.dot(lights_intersection, normalized_to_surface))
                #specular
                camera_intersection = normalize_vector(camera - intersection)

                camera_light = normalize_vector(lights_intersection + camera_intersection)

                illumination += closest_sphere['specular'] * light['specular'] * max(0, np.dot(normalized_to_surface,                                
                camera_light)) ** (closest_sphere['shininess'] / 4)

                #Reflection
                color += reflection * illumination
                reflection *= closest_sphere['reflection']

                ray_origin = shifted_point
                ray_direction = reflected_ray_direction(ray_direction,normalized_to_surface)

            image[i, j] =np.clip(color, 0, 1)
        progress = (i+1)/height*100
        print(f"Progress: {progress:.2f}%", end='\r')
    print("")


def init_screen(resolution):
    """
    set the resolution to low def 300x200, medium def 16:9 720x405, high def 16:9 1920x1080
    """
    match resolution:
        case "300x200":
            width = 300
            height = 200
        case "720x405":
            width = 720
            height = 405
        case "1920x1080":
            width = 1920
            height = 1080
    ratio = float(width) / height  # ratio = image width / image height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # screen defined by 4 numbers: left,top,right,bottom (x coordinate ranges from -1 to 1) (y-coord from -1/ratio to 1/ratio)
    return width,height,screen

def benchmark_render(screen, width, height, spheres, kd_tree, model, nr_of_runs=1):
    """
    Does the benchmarking for the renderer and saves the images in a specific folder with corresponding filenames.
    runs the same test with the same parameter nr_of_runs times and then saves the output image.
    """
    execution_times = []
    output_dir = "benchmark_images"
    os.makedirs(output_dir, exist_ok=True)

    for run in range(nr_of_runs):
        image = np.zeros((height, width, 3))
        start_time = time.time()
        render(screen, width, height, spheres, kd_tree, model, image)
        end_time = time.time()
        execution_times.append(end_time - start_time)

        filename = f"{output_dir}/raytracing_{model}_{width}x{height}_{len(spheres)}spheres_{end_time-start_time:.1f}secs.png"
        print(f"Done with: {model}_{width}x{height}_{len(spheres)}spheres_{end_time-start_time:.1f}secs")
        plt.imsave(filename, image)

    return execution_times


def run_benchmark(resolutions, num_spheres_range, nr_of_runs):
    """
    Runs the benchmarks for different resolutions and numbers of spheres.
    """
    results = {}

    for resolution in resolutions:
        width, height, screen = init_screen(resolution)
        results[resolution] = {}

        for num_spheres in num_spheres_range:
            spheres = create_evenly_spaced_spheres(num_spheres, num_spheres, 0.1, screen)
            sphere_centers = np.array([sphere['center'] for sphere in spheres])
            kd_tree = KDTree(sphere_centers)

            results[resolution][num_spheres] = {}
            results[resolution][num_spheres]['kdtree'] = []
            results[resolution][num_spheres]['naive'] = []

            for model in ["kdtree", "naive"]:
                execution_times = benchmark_render(screen, width, height, spheres, kd_tree if model == "kdtree" else None, model, nr_of_runs)
                results[resolution][num_spheres][model].extend(execution_times)

    return results

def save_results_to_file(results, filename="benchmark_results.txt"):
    """
    Saves the benchmark results into a file and indents it nicesly for good view
    """
    with open(filename, "w") as f:
        for resolution, resolution_data in results.items():
            f.write(f"Resolution: {resolution}\n")
            for num_spheres, models in resolution_data.items():
                f.write(f"  Number of Spheres: {num_spheres}\n")
                for model, times in models.items():
                    f.write(f"    {model}: {times}\n")
            f.write("\n")


def plot_results(results):
    """
    Creates plots of execution time vs. number of spheres for each resolution.
    """
    for resolution, resolution_data in results.items():
        num_spheres_list= list(resolution_data.keys())
        num_spheres_squared_list = [(int(num_spheres) ** 2 + 1) for num_spheres in num_spheres_list]  # Square the values
        kdtree_times = [np.mean(resolution_data[num_spheres]['kdtree']) for num_spheres in num_spheres_list]
        naive_times = [np.mean(resolution_data[num_spheres]['naive']) for num_spheres in num_spheres_list]

        plt.figure()
        plt.plot(num_spheres_squared_list, kdtree_times, label="KDTree")
        plt.plot(num_spheres_squared_list, naive_times, label="Naive")
        plt.xlabel("Number of Spheres")
        plt.ylabel("Execution Time seconds")
        plt.title(f"Execution Time vs. Number of Spheres ({resolution})")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"plot_{resolution}.png")
        plt.show()


def main():

    width,height,screen = init_screen("300x200") #set the resolution to 300x200, 720x405 or 1920x1080
    spheres = create_evenly_spaced_spheres(10,10,0.1,screen)
    sphere_centers = np.array([sphere['center'] for sphere in spheres])
    kd_tree = KDTree(sphere_centers)
    image = np.zeros((height, width, 3))
    model = "naive"
    models = ["naive","kdtree"]
    for model in models:
        start_time = time.time()
        render(screen,width,height,spheres,kd_tree,model,image)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        filename = f"{model}_{len(spheres)}spheres_{width}x{height}_{end_time - start_time:.1f}secs.png"
        print(filename)
        plt.imsave(filename, image)
    #plt.imshow(image) 
    #plt.axis('off')
    #plt.show()

if __name__ == "__main__":

    # TRY TO GENERATE ONE EXAMPLE
    main()

    # COMMENT OUT main() AND REMOVE THE COMMENTS BELOW FOR BENCHMARKING

    # resolutions = ["300x200", "720x405", "1920x1080"], # Try multiple resolutions, takes a long time
    # resolutions = ["1920x1080"] # Only high def
    # num_spheres_range = range(1, 11, 1)  # Adjust as needed for different amounts of spheres
    # nr_of_runs = 1 # how many times each scene should be rendered. if higher than 1, we measure the mean time, only feasible for 300x200 since render times high for other resolutions

    # results = run_benchmark(resolutions, num_spheres_range, nr_of_runs)
    # save_results_to_file(results)
    # plot_results(results)


# SOME EXAMPLE SPHERES TO BE USED WITH MAIN

# spheres = [
#       The original spheres from Omar Aflaks implementation
#      { 'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'Diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
#      { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'Diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
#      { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'Diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
#      { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'Diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
#     # Added som random spheres 
#     { 'center': np.array([0.3, 0.3, -0.7]), 'radius': 0.15, 'ambient': np.array([0.2, 0.1, 0.1]), 'Diffuse': np.array([0.8, 0.2, 0.2]), 'specular': np.array([1, 1, 1]), 'shininess': 50, 'reflection': 0.4 },
#     { 'center': np.array([-0.3, -0.3, -0.8]), 'radius': 0.1, 'ambient': np.array([0.1, 0.1, 0.2]), 'Diffuse': np.array([0.5, 0.5, 0.8]), 'specular': np.array([1, 1, 1]), 'shininess': 70, 'reflection': 0.6 },
#     { 'center': np.array([0.5, -0.5, -0.4]), 'radius': 0.1, 'ambient': np.array([0, 0.2, 0.2]), 'Diffuse': np.array([0.3, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 90, 'reflection': 0.5 },
#     { 'center': np.array([0.1, 0.6, -0.2]), 'radius': 0.1, 'ambient': np.array([0.2, 0.2, 0.2]), 'Diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.3 },
#
#     # Added some more spheres
#     { 'center': np.array([-0.4, 0.4, -0.4]), 'radius': 0.12, 'ambient': np.array([0.2, 0, 0.2]), 'Diffuse': np.array([0.7, 0.1, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.5 },
#     { 'center': np.array([0.6, -0.2, -0.6]), 'radius': 0.1, 'ambient': np.array([0, 0.2, 0.1]), 'Diffuse': np.array([0.4, 0.8, 0.3]), 'specular': np.array([1, 1, 1]), 'shininess': 70, 'reflection': 0.4 },
#     { 'center': np.array([0.0, -0.6, -0.7]), 'radius': 0.14, 'ambient': np.array([0.3, 0.3, 0.0]), 'Diffuse': np.array([0.8, 0.8, 0.2]), 'specular': np.array([1, 1, 1]), 'shininess': 90, 'reflection': 0.6 },
#     { 'center': np.array([-0.2, -0.5, -0.5]), 'radius': 0.1, 'ambient': np.array([0.1, 0.1, 0.3]), 'Diffuse': np.array([0.5, 0.5, 0.9]), 'specular': np.array([1, 1, 1]), 'shininess': 60, 'reflection': 0.3 }
# ]