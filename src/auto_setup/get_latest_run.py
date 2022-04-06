import json

if __name__ == '__main__':
    with open('jobs.json') as json_file:
        jobs = json.load(json_file)
    # Create a list of all run ids with creation time
    completed_jobs = [(job['name'], job['creation_context']['created_at'])
                      for job in jobs
                      if job['status'] == 'Completed']
    # Print last item in the sorted list
    print(sorted(completed_jobs, key=lambda x: x[1])[-1][0])
