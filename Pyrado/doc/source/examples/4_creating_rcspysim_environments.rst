How to create an RcsPySim environment
-------------------------------------

While Pyrado environments are quite easy to implement by inheriting from `SimEnv` or `RealEnv`, the environments
compatible with `Rcs <https://github.com/HRI-EU/Rcs>`_ require an implementation on the Python side as well as on the
C++ side.

**Part 0: Creating the XML specification**

I assume you already have a config file and the assets describing your complete simulation environment. For
information on how to create these Rcs-specific config files, you can either look at `Rcs <https://github.com/HRI-EU/Rcs>`_
or at the other environments in `PATH_TO/SimuRLacra/RcsPySim/config`. Say you created a folder `HelloMichael` in that
directory, containing at least an Rcs graph file called `gHelloMichael.xml`.

**Part 1: RcsPySim**

For implementing your environment in RcsPySim, it is recommended way to start is by inheriting from `ExperimentConfig`
which can be found in `PATH_TO/SimuRLacra/RcsPySim/src/cpp/core`. The subclass, let's call it `ECHelloMichael`, only
needs a .cpp file. It is highly recommended to check out the existing ExperimentConfig files for inspiration. Here is a
basic template:

.. code-block:: c++

    #include "ExperimentConfig.h"
    #include "action/THE_ACTION_MODEL_YOU_WANT.h"
    #include "action/COULD_BE_MULTIPLE.h"
    #include "initState/THE_INIT_SPACE_MODEL_YOU_WANT.h"
    #include "observation/THE_OBSERVATION_MODEL_YOU_WANT.h"
    #include "observation/COULD_BE_MULTIPLE.h"
    #include "physics/PhysicsParameterManager.h"
    #include "physics/PPD_PARAMETER_TO_RANDOMIZE.h"
    #include "physics/COULD_BE_MULTIPLE.h"
    #include "physics/ForceDisturber.h"

    #include <Rcs_INCLUDES_YOU_NEED.h>

    #ifdef GRAPHICS_AVAILABLE
    #include <RcsViewer.h>
    #endif

    #include <OTHER_LIBS>

    namespace Rcs
    {

    class ECHelloMichael : public ExperimentConfig
    {

    protected:
        virtual ActionModel* createActionModel() {
        // TODO Create an action model. 
        }

        virtual ObservationModel* createObservationModel(){
        // TODO Create an observation model. 
        }

        virtual void populatePhysicsParameters(PhysicsParameterManager* manager){
        // TODO (OPTIONAL) Add domain parameters for randomization. 
        }

    public:
        virtual InitStateSetter* createInitStateSetter(){
        // TODO Create an initial state space model. 
        }

        virtual ForceDisturber* createForceDisturber(){
        // TODO (OPTIONAL) Create force a disturber that hinders your agent. 
        }

        virtual void initViewer(Rcs::Viewer* viewer){
    #ifdef GRAPHICS_AVAILABLE
        // TODO (OPTIONAL) Initialize the viewer. 
    #endif
        }

        void
        getHUDText(
            std::vector<std::string>& linesOut,
            double currentTime,
            const MatNd* obs,
            const MatNd* currentAction,
            PhysicsBase* simulator,
            PhysicsParameterManager* physicsManager,
            ForceDisturber* forceDisturber) override{
        // TODO (OPTIONAL) Set the text for the HUD of the viewer.
        }
    
    };

    static ExperimentConfigRegistration<ECHelloMichael> RegHelloMichael("HelloMichael");

    }

**Part 2: Pyrado**

You already made the more difficult part of the implementation. Next, we create the counterpart of `ECHelloMichael` in
`PATH_TO/SimuRLacra/Pyrado/pyrado/environments/rcspysim/hello_michael.py`, by inheriting from `RcsSim` which can be
found in `PATH_TO/SimuRLacra/Pyrado/pyrado/environments/rcspysim/base.py`. Here is a basic template:

.. code-block:: python

    class HelloMichaelSim(RcsSim, Serializable):

        name: str = "hm"

        def __init__(self, task_args: dict, max_dist_force: float = None, **kwargs):
            Serializable._init(self, locals())

            # TODO Forward basic arguments as well as custom arguments to RcsSim's constructor, which will then pass it to the ECHelloMichael.
            RcsSim.__init__(
                self,
                task_args=task_args,
                envType="HelloMichael",
                graphFileName=kwargs.pop("graphFileName", "gHelloMichael.xml"),
                **kwargs,
            )

            self._max_dist_force = max_dist_force

        def _disturbance_generator(self) -> (np.ndarray, None):
            # TODO (OPTIONAL) Generate 3-dim disturbance force. If not used, return `None`.

        @classmethod
        def get_nominal_domain_param(cls) -> dict:
            # TODO Define `dict` with domain parameter names as keys and their parameter 1-dim values as value.

        def _create_task(self, task_args: dict) -> Task:
            # TODO Define a Pyrado Task. There are many possibilities here. In general, the tasks operate on `self.state`.

Inheriting from `Serializable` and calling `Serializable._init(self, locals())` right at the beginning of the constructor
is necessary in order to parallelize the rollouts during training later. The `task_args` are forwarded to the constructor
of `RcsSim`, which is calling `self._create_task()`. The argument `envType` must match the name we passed to the
C++ registry function above, since it selects which `ExperimentConfig` we are using. Furthermore, it is very important
which `graphFileName` we are passing. It is good practice to only use one graph file per `RcsSim` child class, and rather
create a new sibling class of `HelloMichaelSim` and make both inherit from an (abstract) base class `HelloMichaelBaseSim`
which implements the common parts. The keyword arguments are forwarded to `RcsSim` and eventually to the C++ implementation.
Examples for `kwargs` are `checkJointLimits: bool` or `collisionConfig: dict`, but there are many more which can also be
custom to your environment. On the C++ side, these arguments can be retrieved from the member `properties` which exists
for every `ExperimentConfig`. Two examples are `properties->getPropertyBool()` to pass a Boolean or `properties->getChildList()`
to pass a python dictionary.

**Part 3: Run it**

The easiest way to inspect your simulation now is to create a simple script, let's call it `sb_hm.py`, in
`PATH_TO/SimuRLacra/Pyrado/scripts/sandbox`, which creates an instance of `HelloMichaelSim` and runs a idle policy sending
zero actions all the time. In the following snippet, we chose the physics engine to be Bullet, the time step size as 0.01s
and the overall time to 10s. Moreover, we are ignoring joint limits, and have no disturbance.

.. code-block:: python

    import rcsenv
    import pyrado
    from pyrado.environments.rcspysim.hello_michael import HelloMichaelSim
    from pyrado.policies.special.dummy import IdlePolicy
    from pyrado.sampling.rollout import rollout
    from pyrado.utils.data_types import RenderMode


    rcsenv.setLogLevel(4)

    if __name__ == "__main__":
        # Set up environment
        env = HelloMichaelSim(
            physicsEngine="Bullet",
            dt=0.01,
            max_steps=1000,
            max_dist_force=None,
            checkJointLimits=False,
        )

        # Set up policy
        policy = IdlePolicy(env.spec, policy_fcn, dt)

        # Simulate
        return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)

**(optional) Part 4: Run it from C++**

You can also test your environment and policy from C++. This is however a bit more intrigued and still experimental.
First, we need to create an experiment description xml-file. One possible case is that you ran an experiment, i.e.
trained a policy. Next you can go to `cd PATH_TO/SimuRLacra/Pyrado/scripts/deployment`, activate the anaconda environment,
and run `python export_policy_cpp.py`. This will create an `ex_<CUSTOM>_export.xml` as well as an `policy_export.pt` file.
If you want the your trained policy to be played back, you need to insert the line
`<policy type="torch" file="policy_export.pt"/>` into the experiment XML-file. However, this is optional. Actually, you
don't really need to run an experiment to use this export script. Feel free to change it, or generate the XML-file
manually. The important lines in the export script ware are:

.. code-block:: python

    exp_export_file = osp.join(ex_dir, f"ex_{env.name}_export.xml")
    inner_env(env).save_config_xml(exp_export_file)

Next you copy the `ex_<CUSTOM>_export.xml` file (and optionally the policy) to `cd PATH_TO/SimuRLacra/RcsPySim/build`
and then run `bin/TestBotSim -dl 2 -dir ../config/HelloMichael -f ex_<CUSTOM>_export.xml`.