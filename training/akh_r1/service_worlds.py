"""Declared finite sampling spaces, separate DAG and route-allocation mechanisms."""
import random

def generate(seed):
 rng=random.Random(seed)
 n=rng.randint(3,5);nodes=['support-'+str(i) for i in range(n)]
 # Every nonroot has a parent; extra edges vary branching and join structure.
 dag={nodes[0]:[]}
 for i in range(1,n):
  parents={nodes[rng.randrange(i)]}
  parents.update(nodes[j] for j in range(i) if rng.random()<0.4)
  dag[nodes[i]]=sorted(parents)
 count=rng.randint(3,5);occupied=rng.randint(1,count-1)
 routes={'route-'+str(i):('external-owner-'+str(i) if i<occupied else None) for i in range(count)}
 common=dict(authority='nodealpha',active=None,dependencies={},lease=None,probe=None,epoch=1,audit=[])
 return [dict(id='dependency',split='development',mechanism='dependency_start',initial_state=dict(common,mechanism='dependency_start',dag=dag,routes={},route_health={})),dict(id='routing',split='validation',mechanism='drain_switch',initial_state=dict(common,mechanism='drain_switch',dag={},routes=routes,route_health={r:True for r in routes}))]


def generate_breadth_descriptor(split,seed):
 """Explicit decisions rather than a supplied recipe; descriptors, not execution."""
 if split not in ('development','validation'):raise ValueError('Unknown split')
 rng=random.Random(seed)
 if split=='development':
  n=rng.randint(4,6);nodes=['support-'+str(i) for i in range(n)];dag={nodes[0]:[]}
  for i in range(1,n):dag[nodes[i]]=sorted({nodes[rng.randrange(i)]}|({nodes[0]} if i>1 and rng.random()<0.5 else set()))
  affected=nodes[rng.randint(1,n-1)];replacement='replacement-support';revised={k:[replacement if p==affected else p for p in v] for k,v in dag.items() if k!=affected};revised[replacement]=dag[affected][:]
  return dict(schema='akh-service-breadth-v1',split=split,seed=seed,mechanism='support_cut_repair',public_start=dict(goal='Discover current required support and restore service only after it is valid',target='service-a'),initial_observations=dict(dag=dag,source_version=1),examiner=dict(later_trigger='after_initial_verified_start',revoked=affected,later_observations=dict(dag=revised,source_version=2),expected_initial_support=sorted(dag),expected_later_support=sorted(revised)),assistance=['Structured support observations; no topological recipe'],executor_gap='Existing CLI advertises a recipe and has no support revocation operation',qualified=False,new_world_credit=0)
 count=rng.randint(3,5);routes=['route-'+str(i) for i in range(count)];chosen=rng.choice(routes);later=rng.choice([r for r in routes if r!=chosen]);demand=rng.randint(2,4)
 capacities={r:rng.randint(1,4) for r in routes};used={r:capacities[r] for r in routes};capacities[chosen]=demand+rng.randint(0,2);used[chosen]=capacities[chosen]-demand
 revised_capacity=dict(capacities);revised_used=dict(used);revised_used[chosen]=capacities[chosen];revised_capacity[later]=demand+1;revised_used[later]=1
 return dict(schema='akh-service-breadth-v1',split=split,seed=seed,mechanism='route_capacity_transfer',public_start=dict(goal='Discover a healthy route with enough observed capacity; drain dependents before transfer',target='service-a',demand=demand),initial_observations=dict(capacity=capacities,used=used,healthy={r:True for r in routes},dependents=['client-'+str(i) for i in range(rng.randint(1,3))],source_version=1),examiner=dict(later_trigger='after_initial_verified_admission',later_observations=dict(capacity=revised_capacity,used=revised_used,healthy={r:True for r in routes},source_version=2),expected_initial_route=chosen,expected_later_route=later),assistance=['Structured capacity/health observations; candidate selects route'],executor_gap='Existing CLI auto-selects first empty exclusive route; lacks explicit capacity reservation and dependent drain',qualified=False,new_world_credit=0)


def validate_breadth_descriptor(c):
 if c['examiner']['later_trigger'] not in ('after_initial_verified_start','after_initial_verified_admission'):raise ValueError('Missing causal correction')
 if c['split']=='development':
  def acyclic(dag):
   done=set()
   while len(done)<len(dag):
    ready={n for n,parents in dag.items() if set(parents)<=done}-done
    if not ready:raise ValueError('Cycle or undefined support')
    done|=ready
  initial=c['initial_observations']['dag'];later=c['examiner']['later_observations']['dag'];acyclic(initial);acyclic(later)
  if c['examiner']['revoked'] not in initial or c['examiner']['revoked'] in later:raise ValueError('Revocation did not change support')
  if sorted(initial)!=c['examiner']['expected_initial_support'] or sorted(later)!=c['examiner']['expected_later_support']:raise ValueError('Incorrect support expectation')
 else:
  demand=c['public_start']['demand']
  for obs,key in [(c['initial_observations'],'expected_initial_route'),(c['examiner']['later_observations'],'expected_later_route')]:
   feasible=[r for r in obs['capacity'] if obs['healthy'][r] and obs['capacity'][r]-obs['used'][r]>=demand]
   if feasible!=[c['examiner'][key]]:raise ValueError('Route must uniquely satisfy independent capacity/health constraints')
  if c['examiner']['expected_initial_route']==c['examiner']['expected_later_route']:raise ValueError('Correction must change selected route')
 return True


def generate_production_descriptor_v2(split, seed):
 """Unqualified production candidate; legacy descriptors remain unchanged.

 No sampling or execution is performed until the caller invokes this function.
 A separate scorer and lifecycle binding are required before qualification.
 """
 case = generate_breadth_descriptor(split, seed)
 case['schema'] = 'akh-service-breadth-v2'
 case['executor_gap'] = 'Production v2 requires independent scorer and lifecycle binding'
 if split == 'development':
  # Add a prerequisite between existing nodes: edge count changes, so this
  # cannot be an identifier permutation of the initial graph.
  nodes = sorted(case['initial_observations']['dag'])
  root, prerequisite, target = nodes[0], nodes[1], nodes[-1]
  initial = {node: ([] if node == root else [root]) for node in nodes}
  later = {node: list(parents) for node, parents in initial.items()}
  later[target] = [root, prerequisite]
  case['mechanism'] = 'dependency_edge_addition'
  case['initial_observations']['dag'] = initial
  private = case['examiner']
  private.pop('revoked')
  private['later_observations']['dag'] = later
  private['expected_initial_support'] = nodes
  private['expected_later_support'] = nodes
  private['intervention'] = {'dependent': target, 'added_prerequisite': prerequisite}
 else:
  # In each snapshot one healthy route is capacity-limited and another has
  # ample capacity but is unhealthy. Neither predicate alone selects safely.
  private = case['examiner']
  initial_target = private['expected_initial_route']
  later_target = private['expected_later_route']
  distractor = next(route for route in case['initial_observations']['capacity']
                    if route not in (initial_target, later_target))
  demand = case['public_start']['demand']
  for observation in (case['initial_observations'], private['later_observations']):
   observation['capacity'][distractor] = demand + 1
   observation['used'][distractor] = 0
   observation['healthy'][distractor] = False
  case['mechanism'] = 'capacity_and_health_transfer'
 return case
