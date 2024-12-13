import requests
import json

# FHIR server base URL
FHIR_SERVER_URL = "http://localhost:8080/fhir"  # Update this to match your FHIR server's local endpoint

def fetch_resource_by_id(resource_type, resource_id, fhir_server_url):
    """
    Fetches a specific resource by its type and ID from the FHIR server.

    :param resource_type: The type of the FHIR resource (e.g., 'Patient', 'Observation').
    :param resource_id: The ID of the resource to fetch.
    :param fhir_server_url: The base URL of the FHIR server.
    :return: The fetched resource or None if not found.
    """
    url = f"{fhir_server_url}/{resource_type}/{resource_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch {resource_type} with ID {resource_id}.")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def fetch_all_resources(resource_type, fhir_server_url):
    """
    Fetches all resources of a specific type from the FHIR server.

    :param resource_type: The type of the FHIR resource (e.g., 'Patient', 'Observation').
    :param fhir_server_url: The base URL of the FHIR server.
    :return: The list of fetched resources.
    """
    url = f"{fhir_server_url}/{resource_type}"
    response = requests.get(url)
    if response.status_code == 200:
        bundle = response.json()
        if 'entry' in bundle:
            return [entry['resource'] for entry in bundle['entry']]
        else:
            print(f"No {resource_type} resources found.")
            return []
    else:
        print(f"Failed to fetch {resource_type} resources.")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return []

# Example: Fetching specific resources by ID
patient = fetch_resource_by_id("Patient", "P2", FHIR_SERVER_URL)
if patient:
    print(f"Fetched Patient: {json.dumps(patient, indent=2)}")

# Example: Fetching all resources of a specific type
all_patients = fetch_all_resources("Patient", FHIR_SERVER_URL)
if all_patients:
    print(f"Fetched {len(all_patients)} Patients:")
    for patient in all_patients:
        print(json.dumps(patient, indent=2))
