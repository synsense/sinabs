# Release notes

## Unreleased

* Merge Release Notes and Changelog.
    * Remove Changelog file as it was not being used.
* Remove Pipfile and Pipfile.lock files as they were not being used anymore.
    * Specific/minimal library versions can be found in requirements.txt

## v3.0.3 (22/07/2025)

* Add function in utils to help identify issues with memory constraints when mapping a network on Speck.
(https://sinabs.readthedocs.io/v3.0.3/api/utils.html#sinabs.utils.validate_memory_mapping_speck)

## v3.0.2 (10/06/2025)

* Update contact email. Now, any support requests and information about contributors license agreement needs to be sent to `support@synsense.ai`.

## v3.0.1 (06/06/2025)

* Update the release of the project when `main` branch is updated instead of `develop`

## v3.0.0 (06/06/2025)

* Remove support for older boards following update on Samna 0.46.0

## v2.0.3 (18/03/2025)

* Update Sinabs license to Apache 2.0

## v2.0.2 (23/01/2025)

* Spike count plot in 'DynapcnnVisualizer' is optional 
* `DynapcnnVisualizer` allows custom JIT filters to make readout predictions

## v2.0.0 (14/03/2024)

* Fix typos and broken links in the documentation
* Move instructions for adding new device to FAQ
* Move tutorials from speck to top level
* Fix build of documentation with new sphinx version
* Update tests to new API
* Upgrade black to v24.1.0
* Remove ONNX support in favor of NIR
* Add notebooks to ignore execution list
* Fix unit test to check for deprecation warnings correctly
* Include firing rate analysis in synop loss tutorial
* Add deprecation warnings to SNNAnalyzer and SynOpsCouner
* Add missing typehint import
* Add tutorial about nir deployment on speck
* Add NIR api to docs
* Add test for extended readout layer
* Add test for NIR submodule
* Add test for raster to events
* Merge DynapCNN and Sinabs **breaking change**
* Update synop loss tutorials for snns and anns
* Ensure connection map for synop model hook is on same device as input
* Add test for synop model hook on cuda
* Add unit tests for firing rate and input diff hooks
* Speed up synops hook
* Add unit tests for model-to-layer conversion
* Allow ignoring layers when converting to dynapcnn network
* Add unit tests for synops hooks
* Add import to Visualizer back
* Add hook for novel 'diff input' metric
* Remove easy-install command
* Add graphviz dependency

## v1.2.10 (07/12/2023)

* Update codecov.yml
* Update ci-pipeline.yml
* Create codecov.yml
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* Fix wrong variable name
* Update .gitlab-ci.yml file
* Add installation of pytest
* Update docker image to be used in the pipeline
* Add modifications to Authors and ChangeLog
* corrected data formats while loading nir
* added ignore dims to extract nir
* flatten layer dims updated to NIR speck
* added in input\_shape
* Install CPU version of torch for CI pipeline
* Update .readthedocs.yaml
* added method to documents
* added method to change batch size
* lower threshold equal to threshold while loading
* added identity transformations for IO
* removed debug message
* bug fix in stride of sumpool
* added citation to readme file
* added cff file
* set default value to 1 for stride in SumPool2d
* updated nir to support Flatten and SumPool2d

## v1.2.9 (31/08/2023)

* Add unit test to ensure that subract value of MembraneSubtract can be a tensor
* Correct test whether subtract value is None
* add figure for input data flow
* fix confusion doc string of func ChipFactory.xytp\_to\_events about the key of the structured array
* update error in available pooling size
* add api indexing for specksim and visualizer
* add indexing for specksim
* fix typos and invalid url links
* add play with dvs on speck notebook
* Update requirements.txt
* Update ChangeLog to resolve conflict
* Unit test for correct duration of forward pass
* Record for at least the duration of the input
* fix typo in the quick start notebook
* commit jupyter notebook change
* add description for dvs visualization
* add N-MNIST quick start tutorial notebook
* complete contents for training tips section
* add items for device management
* Only publish pre-release when on develop branch
* device management
* fix terminology for samna configuration obj
* fix invalid urls in the docs
* Additional information on the architecture restrictions in the documentation
* sinabs/backend/dynapcnn/specksim.py
* Update the markdown with state monitoring
* Fixed the reset states test
* Changed the forward pass to make sure we receive all the events as fast as possible
* fix warnings related with overview.md can not be found when build doc
* Specksim documentation added
* Add tests to the monitoring
* Change readout and add a delay. (WIP: This will be reworked.)
* Several bugfixes, inclusion of the ability to reset network states and ability to monitor internal events
* Add unit test for conversion from events to raster
* Change order of dimensions in events\_to\_raster output to TxCxWxH
* Rename cpp binding functionality test
* Update gesture\_viz to work with the new structure
* Change the MNIST example to work with the new structure
* Fix rasterization time step
* Add unit test for monitor='all'
* change name from examples/dynapcnn\_visualizer to examples/visualizer update the visualizer documentation
* examples/mnist/specksim\_network.py
* Restructure the MNIST example
* Restructure examples
* Move the test specksim network under examples
* Divide examples into subfolders
* Support DynapcnnNetwork conversion. Fix the bugs related to AvgPool2d conversion
* Fix bug when monitor=all and dvs layer is present. Clarify some docstrings
* Bugfixes based on the MNIST script. Still not seeing any output
* Implementation and API is complete. Not tested yet
* Initial commit on the structure
* Namespace change on the tests
* add output monitoring introction for FAQs
* add example of implementing Residual Block
* add example of implementing Residual Block
* reform the FAQ section of the doc
* Tests related to specksim cpp bindings are implemented and added
* deal with v\_leak / r of size 0
* use nirtorch.load and use str indexing where possible
* added Conv2d to NIR
* remove superfluous comments
* use deprecated torch.testing.assert\_allclose to make tests for older pytorch pass
* add nir IF support
* updated Sinabs to latest NIR

## 1.2.8 (10/07/2023)

* add nir and nirtorch to requirements
* implemented sequential for from\_nir
* add test for to\_nir
* deleted old files
* added channel shift layer
* removed graph (migrated to nirtorch)
* Add missing nodes
* minor fixes in node conversion
* wip: restructure code
* wip: continue import from nir
* wip conversion from nir to sinabs
* Fix issue #99, add unit test

## v1.2.6 (30/06/2023)

* change nmnist tutorial title
* added NMNIST tutorial
* modified Add to Morph
* function call bug fix in ignore\_nodes
* back to previous version
* back to previous code
* refactor code, change the extended method from encrypted function to extra function
* fixed variable assignment issue
* modify the documentation for extend method
* fixed some typo
* remove unneeded package importing
* add readout layer extend for Speck2E and 2F
* update ChangeLog
* blackened
* added method to ignore tensors in graph
* Unit test for periodic exponential gradient
* Generate periodic exponential surrogate gradient correctly for different thresholds. Solves issue #97
* removed torchvision from test requirements
* using cat instead of concat
* added doc string
* added assert
* balckened
* added optional model name to ignore it from graph
* added test for branched SNN
* added modules for addition and concatenation
* added convenience method for graph extraction
* removed torchview imports
* added sensible tests conditions in place of prints
* only saving index of last used tensor id
* Added methods to simplify the graph
* updated graph definition
* removing redundant incoming nodes attribute
* added context manager
* wip. basic graph exteaction and tracing added
* Warning about running the script for MacOS users added to the documentation
* minor changes and corrections
* Updated info on biases/leak
* added how to list connected devices
* Fix plot axes labels
* Removed commented kwargs from methods
* updated files
* Add unit test for DynapcnnNetwork.is\_compatible\_with
* Bugfix in DynapcnnNetwork.is\_compatible\_with where it would raise a ValueError instead of returning False
* Added documentaion in FAQ
* added is\_compatible\_with method; blackened
* updated DynapcnnVisualizer connect method
* moved configuration settings order
* blacked and updaetd test
* updated dynapcnn visualizer with latest samna api
* wip, dynapcnn visualizer refractoring
* wip updating dynapcnn visualizer
* blackened; updated to use run\_vusualizer method
* Added method for launching samna visualizer
* Updated notebook to the new samna graph api
* complete the neuron-leak example notebook
* add leak-neuron link in the FAQs
* merge develop to 76-add-leak-neruon-option
* fixed links to static images in notebooks
* fixed links
* shuffled the order of documentation
* setup file params use \_ instead of - now
* Fixed link to samna
* Make sure DVSLayer is added to DynapcnnNetwork if and only if necessary. Fix unit test about single neuron on hardware
* Turn off DVS sensor when DynapcnnNetwork has no DVSLayer
* Restructure unit tests
* Fix: Allow input shape to DVS layer to have 2 channels
* More flexible DVSLayer generation
* Remove 'compatible\_layers' attribute from DynapcnnNetwork and use 'sequence' instead
* documentation improvement
* replace curl with wget
* update ChangeLog
* replace all math equation
* Added board resetting before second test. Note that the device will be opened already before the second test starts, so it is not actually necessary
* Implemented resetting the boards before running the test
* add getting started with speck section
* Add none type comparison to spike\_threshold and min\_v\_mem
* Updated dangers.md with suggestion of using MultiSpike for training models to be deployed to chip
* skip visualizer test
* warn if layer with pooling is monitored
* add speck image at the doc homepage
* Add notes in docstrings and in the documentation about the fact that attribute modifications after the  call have no effects
* specify inline math latex for Overview tutorial
* make tutorial notebook that contains latex a markdown documentsince it doesn't contain any code
* add skipif condition for test-case that need hardware connected to the CI machine
* update changelog
* add neuron leak tutorial notebook
* fix bug for neuron address mapping and add unit test case
* update ChangeLog
* add neruon leak option and related unit test
* remove system package install cmd
* fix add-apt-repository can't be found error
* add apt repository for graphviz localization
* try another way install graphviz
* fix CI pipeline failure

## v1.2.5 (24/04/2023)

* make contact title bold like the others
* add contact section to documentation
* checked tests pass on mac

## v1.2.4 (05/04/2023)

* replace torch.inf with math.inf
* ignore data and cache directories
* added test
* Take averace across samples within batch when collecting synops
* Unit test for synops counter with batch size > 1
* added noconfirm
* added update keyring to gitlab-ci
* renamed test

## v1.2.3

* fix synops/s for Linear layers when no Squeeze layers are used in network
* update pre-commit config

## v1.2.2 (17/03/2023)

* expand saved input in spiking layers for analyzer
* derive spiking layer stats from single saved output tensor and also save input
* Speck2fDevKit support added
* add n\_neurons to model\_stats
* Test added
* Create random indices tensor on the same device as the buffer
* update samna version requirements
* update ChangeLog
* add speck2f module devkit support and remove speck2f characterization board support

## v1.2.1 (23/02/2023)

* distinguish between accumulated and mini batch stats for firing rates
* distinguish between accumulated and mini batch statistics in SNNAnalyzer for synops
* only compute connection\_map once in SynopsHook
* detach synops accumulation inbetween mini batches for SNNAnalyzer

## v1.2.0 (15/02/2023)

Included the SNNAnalyzer module to collect model statistics such as number of synops or neurons automatically.

* update release notes
* make sure deconvolve outputs same size as conv input in SNNAnalyzer
* Implementation a test that will create a single neuron on a single core, that receives a single event and outputs a single event
* edit visualizer.md
* Added the visualizer documentation to the index.rst
* Replacing the notebook which cannot spawn a samnagui window with a markdown file
* Visualizer example updated with the window scale parameter
* Window scale parameter is now passed to the constructor
* load model onto CPU and change warning message for JIT compilation
* exclude Samna visualiser generated file
* Samna version requirement incremented to cover the latest changes
* Notebook forgotten connect method added. Readout visualizer only supports .png images. Information regarding to this has been added to the documentation
* Example notebook with explanations added
* Minor bugfixes in readout layout and feature name automatic setting to numbers
* A fully working model
* (WIP) Apart from one issue on not getting any output from the chip it works
* make it possible to use IAF with from\_model
* (WIP) blocked by problems in samna. Turns out  and  were not tested in samna prior to release. Sys-Int team will be fixing it, then I can fix  in samna. Then all the features, but plotting the readoutpins will function
* Tests updated to cover the values, so that we can confirm that the states indeed have not been changed but updated
* remove outdated references to SpkConverter and SNNSynopCounter
* Clarify scaling factor
* clarify in docstring why we're using transpose convolution and scaling factor for AvgPool layer in SNNAnalyzer
* add tutorial text about how to scale weights for conversion
* Clarify in-code documentation
* Add test for changing the trailing dim, which should not happen
* Add documentation to the mismatch handling method
* For non-squeeze layers in the case of input with different batch sizes are received, the states are going to be reshaped accordingly. The change will be randomly sampling states from created by samples in the existing batch repeatedly
* Test norm\_input for LIF
* Test norm\_input for ExpLeak
* added docstring description for SNNAnalyzer
* fix errors in using\_radout\_layer.ipynb
* merge conflicts
* update ChangeLog
* modify config making tests can cover different types of devkits
* add speck2f characterization board support
* Waiting for sys-int team to provide a JitDvsToViz template directly into samna itself. I implement it myself if it is not ready on monday. Anything related to readout has been removed, as well as the JitReadoutNode. This functionality already exists in  under . The API that uses strings in the front end to create plots have been changed with one that uses boolean flags to create readout and power monitor plots
* fix errors in tests after init DVSLayer in any case
* make DVSLayer can be initnitialized in anycase
* (WIP) For readout we have been using something that has been out of performance, this should be replaced with a proper functionality. In order to support all the chips and cameras ToViz converter filters should be Jitted, for which there is no template in  right now
* (WIP) Dynapcnn Visualizer initial implementation
* black formatted
* renamed test
* update ChangeLog
* update figure
* add a time.sleep statement after dynapcnn network reset-states to wait the config take effect
* modify speck2e test to deploy dummy network
* added test for killbit
* test updated to check for type
* added checks if the network is empty
* add spike count visualization notebook into tutorials.rst
* fix typo in the notebook of visualize spike count
* delete chip top level figure
* change the speck top level figure path
* change top level figure
* add notebooks into docs/source/tutorials.rst
* update the noteboob
* Update requirements.txt
* notebook and image for spike counts visualization
* modify the parent folder name identical to the notebook
* modify the image load path
* make spike\_threshold and min\_v\_mem a parameter
* fix test which seems to have relied on older version of PyTorch
* fix wrong calculation formula in remapping neruon index function
* use deconvolve module to compute connection map for accurate synops
* fix conflict when merge develop into 34-add-readout-layer-usage-instruction-notebook
* add samna graph plot for power monitor notebook
* add notebook for power monitoring
* update the readout-layer intro notebook
* add FAQ and known dev kit bugs
* rm change in tutorials.rst to avoid merge conflict
* add seaborn to docs requirements
* replace plotly image with seaborn
* update synops\_loss\_ann notebook
* use consistent American spelling of analyzer

## v1.1.6 (08/12/2022)

* make min\_vem and spike\_thresholds parameters do that they're included in state\_dict
* Detach recordings before plotting
* Allow backpropagating through recorded states
* add figures for overview notebook
* add overview tutorial notebook for speck/dynapcnn
* removed all samna checks from imports
* imports fixed and type annotation for Tuple corrected
* add notebook about dvs input visualization
* added standardize\_device\_id before all IO methods
* add codecov badges
* pytest command fixed for coverage
* added path to installation for coverage
* try hard-coded path for coverage
* added conditional deplopment
* rerunning test in deploy stage
* custom stage added
* added dependenct to unitttests
* two stage coverage deployment
* changed command for acquiring code coverage
* coverage path fixed
* remove gallery from docs as not needed at the moment
* added galary template, updated config -> sinabs
* add coverage config file that specifies the folder to include in the coverage, excluding the test folder
* remove some whitespace in the gitlab ci config
* use codecov token so that hopefully CI is pushed via gitlab.com
* update gitlab CI to produce html coverage report for pages
* try using gitlab pages for coverage report
* adding test coverage job to CI pipeline according to Python template from gitlab.com
* remove gallery from docs as not needed at the moment
* add figures for the readout layer intro notebook
* add readout layer usage example notebook
* added galary template, updated config -> sinabs
* init tutorial file
* Fix to the issue
* Change last uint
* Remove comment
* Uint64 to int
* Check the timestamp and delay factor for negative values
* Casting to int after conversion
* Do not sleep while resetting the states when the function is called. Stop the input graph, while the neuron values are written. Sleep for a bit and then start the graph again
* Fix to the bug

## v1.1.5 (25/11/2022)

* Fix backward pass for MaxSpike
* update ChangeLog after add support of speck2e
* add 'spiking' and 'parameter' subdictionaries to layer stats in snn analyzer
* add reset method for analyzer
* SNNAnalyzer able of tracking firing rates across batches
* make sure synops are accumulated across batches
* add speck2e devkit and testboard support
* remove redundant comment code
* Device input graph stopped on destruction of the DynapcnnNetwork object
* Writing all v mem to zeros changed. -> Now when you call this method it creates a temporary buffer and a temporary graph, starts the graph, writes the events, stops the graph then it is destroyed
* Implement an input buffer for all devices: -> In the config\_builder there is now a function called get\_input\_buffer which gives a source node. -> For each chip this is implemented to get the appropriate source node. -> To method changed to construct the input buffer. -> Forward method changed to write the events to this source node
* improve the plotting of which metrics are tracked in the tutorial notebook
* add a plot for firing rate histograms in the snn synops tutorial
* try using reshape instead of unflatten to be compatible with pytorch 1.8
* attempt to make it pytorch 1.8 compatible and fix synops training tutorial
* replace SNNSynopCounter with SNNAnalyser, which calculates statistics for both param and spiking layers
* Name added to io
* naive copy of config generation
* moved kill bits to a separate method
* Type definition updated to Tuple
* compatibility with torch 1.13
* Name added to io
* Failing unit test
* Requirements for samna version updated
* Change to get the same functionality in 'to()' method in samna Version: 0.21.2.0

## v1.1.4 (18/11/2022)

* autoformat all docstrings using docformatter
* fix bug where added\_spike\_output is a ReLU
* Update README.md
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* Update ci-pipeline.yml

## v1.1.3 (17/11/2022)

* Update ci-pipeline.yml
* add generic reset\_states and zero\_grad utility functions

## v1.1.2 (16/11/2022)

* make sure that add\_spike\_output doesn't add spiking layer in original ANN
* removed unused imports
* sort all imports automatically
* moved kill bits to a separate method
* Enable top-level conversion of sinabs layers with 'replace\_module' function. Resolves issue #60
* Enable converting sinabs layers with
* add documentation about how to release a new Sinabs version
* Hotfix new arg\_dict

## v1.1.1 (02/11/2022)

* add arg\_dict property to StatefulLayer
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* bump Python version under test to 3.10
* get rid of exodus import in test\_conversion
* deprecated 'backend' parameter in from\_torch
* add a replace\_module function that will replace specific layer according to a mapper\_fn. Reworked from\_torch
* requirement bumped to 1.0.7

## v1.0.7 (27/10/2022)

* check for pytorch version under test and either call testing.assert\_close or assert\_allclose depending on the version
* using torch.equal instead of np equal
* tests fixed
* fixing test

## v1.0.6

* added zero\_grad to network
* wip enable learning
* set spike thresholds in from\_torch to tensors by default, get rid of torch testing FutureWarning
* convert spike\_threshold to tensor if a float in the constructor
* make spike\_treshold a tensor instead of a float by default
* added zero\_grad function
* updates to latest sinabs requirements

## v1.0.5 (21/10/2022)

* add utils functions to documentation
* update tutorial notebooks to use batch\_size or num\_timesteps for from\_model
* set default\_factory for SpkConverter dataclass
* undo default batch\_size of 1 for from\_torch.from\_model
* exclude samna log files
* get rid of test warnings: Dropout instead of Dropout2d, no autograd fn instantiation, torch.arange

## v1.0.4 (05/10/2022)

* more docstring updates for layers plus cross-references in API documentation
* add shape and attributes to layer docstrings
* Layer docstring updates, now including constructor type hints automatically
* fix MultiGaussian surrogate gradient and add Gaussian surrogate gradient function
* Update README.md
* add Repeat auxiliary layer
* Update ci-pipeline.yml

## v1.0.3 (28/09/2022)

* exclude generated authors / changelog files
* Removed pandas dependency. Adjusted tests accordingly
* add speck2e config builder

## v1.0.2 (14/09/2022)

* additional minor docstring update
* update some more docstrings for activation modules
* doc strings updated

## v1.0.1 (26/08/2022)

Mostly making v0.3.x stable, with a revamped documentation that includes a gallery and how-tos!
* add release notes for v1.0.1

## v1.0.0

* add complete tutorial notebooks since their execution is deactivated on the docs server
* exclude some notebooks from automatic documentation build because they take very long
* update documentation with how tos and gallery
* add matplotlib to docs requirements
* added sphinx gallery to docs requirements
* blacken whole package
* add documentation autosummaries for layers and activations. Small docstring modifications
* first version of Sinabs gallery instead of tutorial notebook that plots neuron models
* added pre-commit hooks for black

## v0.3.5 (18/08/2022)

* Bump stable PyTorch version from 1.10 to 1.12 in CI pipeline
* Fix bug in tutorial due to API update
* Update README.md
* Update README.md
* Fix handling of non-scalar tau\_syn
* Prevent non-integer arguments to Squeeze class from breaking code

## v0.3.4 (21/06/2022)

* Fix critical bug in LIF synaptic forward dynamics
* re-naming from BufferSinkNode to BasicSinkNode
* add unit test for speckmini chips config building
* override get\_default\_config method for speck2dmini config builder
* add config builder for speck2dmini
* override get constraints method for speck2cmini device
* added speck2cmini to the chipfactory
* added speck2cmini configbuilder
* moved config dict generation to builder
* Prevent in-place updates of already recorded state
* hotfix in IAF.\_param\_dict
* Bugfix in param\_dict and unit test
* Unit test and bugfix
* Include . Don't change device when converting between tau and alpha
* make UnflattenTime also work for Pytorch 1.8
* Requirements updated to samna version 0.14.19
* Get rid of get\_opened\_device and its uses
* Update from\_torch.py
* marked test skip
* reuse get\_opened\_device in open\_device
* make sure that moving a network to an opened device doesn't cause an error
* test without re-opening
* update samna install instructions
* also pass randomize arg to spiking layer reset\_states()
* change test device to speck2b in large\_net test
* update tutorial notebook with reset\_states() and fix small bug in config builder
* event constructor modified
* typo in event name
* added sleep time after reset
* Theme updated
* added set\_all\_v\_mem\_to\_zeros method to erase all of vmem memory
* docs updated
* chip\_layer\_ordering only accepts cnn core layers and not the dvs layer
* changed default flag to false
* added test for PeriodicExponential
* added to init file
* added periodic exponential method
* add output normalisation to chip deployment tutorial to make it work much better
* using non-cached property as otherwise failing for Python 3.7
* Remove tau\_mem as a parameter from IAF
* from\_model takes same parameters as IAF

## v0.3.3 (20/04/2022)

* update SNN synops tutorial
* make SNNSynopsCounter work as a loss function
* add first version of synops counter tutorial notebook
* additional parameter added to reset network
* additional parameter added to reset stateful layer
* fixes state recording issue in ExpLeak
* Update README.md
* moved parse\_device\_string into utils. No need of samna to run other modules
* added speck2b tiny support

## v0.3.2 (314/03/2022)

* Rename remaining threshold arguments to spike\_threshold for more consistency
* Update .gitlab-ci.yml
* Update jupyterlab-requirements.txt

## v0.3.1 (23/03/2022)

* Update ci-pipeline.yml
* Update requirements.txt

## v0.3.0 (22/03/2022)

This is a major overhaul which rewrites a large part of the package. 

* Addition of leaky models such as Leaky Integrate and Fire (LIF), Exponential Leaky (ExpLeak) and Adaptive LIF (ALIF).
* Activation module: from sinabs.activation you'll now be able to pick and choose different spike generation, reset mechanism and surrogate gradient functions. You can pass them to the neuron model (LIF, IAF, ...) of your liking if you want to alter the default behavior.
* new documentation on readthedocs
* SpikingLayer has been renamed to IAF (Integrate and Fire).
* State variable names changed and new ones have been added: 'state' is now called 'v_mem' and 'i_syn' is added for neuron layers that use tau_syn.
* New neuron features: support for recurrent connections, recording internal states, normalising inputs by taus, initialisation with shape and more.
* We moved our repo to Github and changed the CI pipeline to Github actions.

* add basic parameter printing in \_\_repr\_\_
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* update layer docstrings and release notes
* Notebook updated for the new version
* rasterize method accumulates multiple spikes within a time step
* bug fix
* added optional size parameter to events\_to\_raster
* Updated to changes in sinabs 0.3
* add new record\_states feature
* small update to activations tutorial notebook
* update tutorial notebooks
* change functional ALIF behaviour so that v\_mem is not reset beneath 0 after a spike
* update neuron\_model plots
* add tutorial about activations
* remove ActivationFunction class and split into separate parameters spike\_fn, reset\_fn and surrogate\_grad\_fn in all layers
* update neuron\_models notebook
* make tau\_syn in IAF more generic and turn off grads for tau\_mem in IAF
* fix warnings about redundant docstrings in sphinx
* blacken whole repo
* refactor activation module
* reintroduce does\_spike property
* renamed threshold\_low to min\_v\_mem
* make IAF inherit directly from LIF
* Update README.md
* fix some imports
* tutorial notebook that plots different neuron models
* update ExpLeak neuron
* remove does\_spike and change default representation
* make ExpLeak directly inherit from LIF with activation\_fn=None
* change default surrogate gradient fn to SingleExponential
* move SqueezeMixin class to reshape.py
* change MNIST class names in tutorials so that they point to same data. Prevent multiple download on RTD server
* update documentation
* exclude dist from git
* Update README.md
* Update README.md
* Notebook updated with the outputs
* bug fixes for inclusion of threshold\_low
* added threshold\_low for IAF and LIF and corresponding test
* added samna.log files to git ignore
* Notebook with new API verified. Still needs to be rendered with dev-kit
* Moved requirements for sphinx
* Removed InputLayer
* Implemented reset states method
* bumped min version for bug fixes
* added logo with white background
* fundamentals added and notebooks fixed with new api
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* updated training with bptt section
* wip
* Update ci-pipeline.yml
* add link to Sinabs-DynapCNN
* show version number in title
* minimum samna version updated in requirements file
* removed extra-index-url
* Update ci-pipeline.yml
* update layer api description in docs
* remove input layer and regroup pooling and to\_spike layers
* update sphinx config
* update about info
* Delete .gitlab-ci.yml
* Delete CONTRIBUTING.md
* Delete CHANGELOG.md
* Update .readthedocs.yaml
* Update .readthedocs.yaml
* update quickstart notebook
* Update README.md
* Update ci-pipeline.yml
* Update requirements.txt
* Update ci-pipeline.yml
* Update ci-pipeline.yml
* move requirements for test
* first version of ci pipeline script
* update gitlab ci script
* blacken some layers
* add parameter norm\_input to LIF layer, re-use lif functional forward call in IAF layers with alpha=1.0, add firing\_rate to spiking layers
* minor changes to activation docs
* add convenience FlattenTime / UnflattenTime layers
* rework weight transfer tutorial
* various docs updates, refurbishing install page, adding differences page,..
* layer docstring updates
* docs api update
* more docs file restructuring
* moving files around in the docs folder
* added new sinabs logo done by @DylanMuir
* moved files in doc/ up one level
* Unit tests for copying uninitialized layers. Makes sure that #25 is resolved
* Add 'does\_spike' property
* Fix float precision issue
* Remove backend conversion
* Unify param\_dict shape entry
* Make sure tau is converted to float
* Rename to exodus
* Add MaxSpike activation
* Make sure tau\_leak is converted to float
* remove deprecated layers
* blackened tests and added one test for multiple taus in LIF
* Minor: -> Efficiency improvement
* fix previous commit
* Scale exp surr. grad width with threshold
* Matching definition of exponential surrogate gradient with slayer
* Standardize API for all io functions
* Modules completed
* minor change in variable name
* Exponential surrogate gradient
* Samna 0.11 support initial commit: -> This version is tested and works, however there are still improvements that can be done
* wip, moved reset\_states to config builder
* remove the use of UninitializedBuffer because introduced in PyTorch 1.9 and therefore not compatible with PyTorch LTS (long term support) v1.8.1
* tau\_mem for LIF neurons is now always calculated on CPU and transferred to original device, for better numerical comparison with SLAYER LIF layer
* only zero gradients if state is initialised

## v0.2.1 (22/02/2022)

* TorchLayer renamed to Layer
    * Added a depricated class TorchLayer with warning
* Added some new layer types
    * BPTT enabled spike activation layer
    * SpikingTemporalConv1dLayer

* added pip install
* added requirements
* Fixed path to conf.py in readthedocs config file
* added rtd yaml file
* split spike detection from reset mechanism in ALIF to be compatible with LSNN paper
* update docstrings
* remove changes introduced by git rebase
* fix bug for reset\_state does not set neuron states to zeros
* add support for auto merge polarity according to inputshape in make\_config
* removed activation state from dynapcnn layer
* add new squeeze layer mixin
* make ALIF membrane decay test pass
* Remove debugging print statement
* Fix discretization unit test
* bug fixes
* change update order in ALIF
* add grad\_scale to MultiGaussian surrogate grad function
* add missing import for MultiGaussian surrogate grad function
* update SingleSpike mechanism to spike when v\_mem >= threshold instead of >
* update ALIF activation function attribute
* checking tol offset, wip
* initialise threshold state as b0 for ALIF classes
* fix init\_states when 2 inputs of different shapes are supplied
* refactor and test reset\_states function
* return copy of states in reset mechanisms rather than in-place modification
* updated state->v\_mem and threshold->activation\_fn.spike\_threshold
* replace state dict that is passed to torch.autograd function and also test backward passes
* fix issue with MembraneReset mechanism
* added tests for initialisation with specific shape
* when resetting, reset on the right device
* merge dev branch back into feature branch
* properly separate layer business from activation function thresholds
* fix device issues with recurrent layers
* revert back to an additional ALIFActivationFunction that passes state['threshold'] instead of self.spike\_threshold to spike\_fn
* remove custom ALIF spike generation function and move threshold state to ActivationFunction.spike\_threshold
* deactivate ONNX tests for now
* fix error where state would not be initialised on the right device
* make network class tests pass
* make backend tests pass
* remove class factories for recurrency and Squeezing completely and just use nn.Flatten and Unflatten in the future
* move Quantize functions to activation module
* update leaky layers and make all copy tests pass
* make LIFRecurrent inherit from LIF
* add Squeeze Module instead of constructing squeeze classes for every layer
* remove debugging print statement
* include LIFRecurrent module and functional forward call for recurrent layers
* update deepcopy method
* update docstrings for activations and leaky layers
* refactor IAF layers
* add MultiGaussian surr grad fn
* update the ResetMechanism docstrings
* refactor ALIF layer
* ALIF refactoring WIP
* remove old threshold API and traces of SpikingLayer
* make reset functions classes
* rename InputLayer file
* delete SpikingLayer
* refactored ExpLeak layer
* remove Activation class and now have option to specify different surrogate gradients
* rename states to state in LIF
* can initialise shape
* break apart activation function into separate spike\_fn, reset\_fn and surrogate\_grad\_fn
* fix initialisation of states if input size changes
* Enable changing backends for ExpLayer
* use a functional forward method in LIF
* minor change in update to i\_syn
* make deepcopy work with weird i\_syn no\_grad exception
* refactoring together with Sadique
* include activation functions for reset and subtract + forward pass tests for those
* tau\_syn can now be None instead of empty parameter
* can now choose to train alphas or not
* update lif tests
* first stab at breaking out activation function
* add support for auto merge polarity according to inputshape in make\_config
* Fixes issues in to\_backend methods if a backend requires a specific device
* Fixes issues in to\_backend methods if a backend requires a specific device
* Address issue #17 and fix some minor issues with torch.exp
* Minor bugfix on events. Int was not propagated when converting dvs events to samna events
* Added a delay factor in seconds so that the first events timestamp is larger than 0
* bug fixes with deep copy in dev
* fix \_param\_dict for pack\_dims.py
* wip
* use list of output spikes that is stacked at the end rather than reserving memory upfront
* update ALIF execution order
* update documentation for leaky layers
* recurrent ALIF layers
* modify order of threshold update update, then spiking within one time step in LIF neuron
* use taus instead of alphas for leaky layers
* change state variable to v\_mem
* Fix default window in threshold functions
* Remove unnecessary line of code in iaf\_bptt.py
* Lift unwanted strict dependency on sinabs-slayer
* to\_backend method in network class
* Alif deepcopy works
* Add unit tests for switching backends and for deepcopying
* Switching between backends works now, even if other backend has not been imported
* update Quantization and Thresholding tools docs
* fixed tests
* minor documentation update
* replacing instances of SpikingLayer to IAF
* deepcopy now works; \_param\_dict -> get\_neuron\_params() LIF added to \_\_init\_\_ file
* Add StatefulLayer (WIP)
* replacing instances of SpikingLayer to IAF
* added monitor layers documentation to the to method as well
* update recurrent module to make it a class factory, which can be combined with Squeeze layers
* renamed LSNN layer back again to ALIF but keep Bellec implementation
* Raster to events without limiting
* Documentation added
* reset states method refractored
* black formatted some layers
* add RecurrentModule to abstract away recurrent layers
* update LSNN layer
* update recurrent LIF layer
* remove ABC from SpikingLayer
* rename ALIF to LSNN layer
* solve timestamp reset
* minimum of torch 1.8 for torch.div with rounding\_mode param
* update leaky layers and their tests
* fix tests
* Forward pass safer implementation
* Macro for easily monitoring all layers
* Synops support for average pooling layers -> Synopcounter now works correctly when average pooling layers are used
* remove dvs\_layer.monitor\_sensor\_enable = True
* divide threshold adaptation by tau\_threshold to isolate the effect of time constants and not current over time
* replace tau parameters such as tau\_mem and tau\_threshold with respective alpha versions
* fix ci
* bug fix and method renated to reset\_states
* Added partial reset method
* squash warning message about floor division by changing to recommended method
* update LeakyExp tests
* update docstring in LIF layer
* fix ExpLeak layer + lif/alif tests
* rename input tensor for gpu
* re-add unit-test for normalize\_weights
* add GPU tensor support for normalize\_weights method
* no more parameter overrides for alif and lif neurons
* zero grad tests for ALIF/LIF and replace in-place operation
* update LIF and ALIF docstrings
* add tests for LIF/ALIF current integration, membrane decay and threshold decay
* remove a wrong condition expression
* add chip\_layers\_ordering checking in make\_config method
* inelegant solution by adjusting the list comprehension in line#252
* Typos and discord community url fix
* Added samna requirements to gitlab ci script
* update LIF and ALIF documentation
* rename spike\_threshold to resting\_threshold
* update Quantize, StochasticRounding to fix Pytorch warning
* replace instantiated ThresholdReset autograd methods with static call, as recommended by pytorch
* lif and alif layer update
* ALIF: reuse LIF forward call and just change some of the functions that are called from it
* reuse detect\_spikes function in ALIF layer
* add initial version of adaptive LIF layer
* rework LIF layer and add first tests for it
* specify threshold type as tensor
* skeleton code
* add a few more lines on the cosmetic name change in release history
* add change log to documentation
* inelegant solution by adjusting the list comprehension in line#252
* update gitignore to exclude MNIST dataset
* update documentation and remnant methods to update DynapcnnCompatibleNetwork to DynapcnnNetwork
* update tutorial notebook
* add DynapcnnCompatibleNetwork to be backwards compatible
* add dt to events\_to\_raster
* change output format of DynapcnnNetwork to tensors
* update filenames and module paths for dynapcnn\_network and dvs\_layer
* Typos and discord community url fix
* Updates and fixes
* Added discord and documentation urls
* tutorial notebook updated
* added tests for monitoring
* test for cropping fixed + samna requirement bump
* DVSLayer.from\_layers take an input of len 3 Added checks for input\_shape
* ci updated to not wait for confirmation
* replaced swapaxes with transpose for backward compatibility with pytorch 1.7
* gitlab ci updated to install latest version of samna
* added doc strings
* Added instructions for how to add support for a new chip
* api docs nested
* wip
* deleted mnist\_speck2b example script as dynapcnn\_devkit works by just replacing the device name
* update API doc
* Default monitor enabled for last layer is nothing is specified
* merged changes
* Removed redundant/legacy code
* rename API doc headings
* Update unit tests according to recent commits
* clean up API documentation by not displaying module names
* Minor fixes and adaptations. More specific exception type. Can pass network with dvs layer to dynapcnn compatible network
* Smaller fixes in config dicts
* Refactored dvs unit tests
* fixed typos in documentation
* Bug fix in crop2d layer handling
* Added Crop2d layer
* installation instructions and minor documentation changes
* Minor changes
* added some folders to gitignore
* moved event generation methods to ChipFactory
* depricated methods deleted from source
* supported\_devices in ChipFactory and get\_output\_buffer in ChipBuilder
* added support for time-stamp management
* enable pixel array on dvs\_input true
* adding speck2b device names + mnist example script
* speck2b bug fix in builder
* removed factory setting line
* added speck2b to the condition
* added speck2b to the condition
* added parameter file for example
* Added config builders for speck and speck2b
* Refractored to add ConfigBuilder
* Support for InputLayer. Still does not pass \`test\_dvs\_input\` tests
* added index for samna
* Cut dimensions update in the configuration dict
* Minor api corrections
* dynapcnn layers populaiton works. Bug in dvs layer still to be sorted out
* wip: build full network
* construct dvs layer construction works
* Added tests for DVSLayer
* Added custom exceptions and tests
* method to build network fo dynapcnn layers added
* added start layer index to construction methods
* Added tests for layer builders
* Added function to create dynapcnn layers
* DVSLayer, FlipDims functional code added
* Suggestion: DVSLayer. Still to be completed
* WIP
* Added handling of sumpool layers at the start of the model
* Updated MNIST example notebook in the documentaion
* added speck2\_constraints
* make\_config default for chip\_layers\_ordering changed to "auto"
* unhide chip\_layers\_ordering
* Breaking change: monitor\_layers now takes model index instead of chip layer index
* wip
* Added API docs for new files
* Added the basic documentation
* minor documentation typo fixes and some clarifications
* doc skeleton added for the fundamentals
* mapping logic updated to edmond algorithm
* Unit test for from\_torch with num\_timesteps
* Added test to check on initialization with batch\_size
* wip
* Slight refactoring: More methods in SpikingLayer
* Fix zero\_grad test
* Test new zero\_grad method
* Added generic zero\_grad method to SpikingLayer class
* override zero\_grad instead of separate method detach\_state\_grad
* Add unit test. Rename detach\_state\_grads to detach\_state\_grad for consistency with no\_grad
* Method for detaching state gradients without resetting
* Random reset into sensible value range
* Fix output shape
* Do not transpose data in IAF.forward
* Remove Squeeze/Unsqueeze helper classes
* Add missing spiking\_layer module. Minor renaming of squeeze classes
* Make sure that Squeeze layers are registered as subclasses of Squeeze class
* Change data format of iaf input: Batch dimension is first. Always implicitly expect a batch dimension
* IAF expects batch and time separated. IAFSqueeze for old behavior with squeezed dimensions
* bug fix in make\_config effecting auto-mapping
* move name\_list acquiring from plot\_comparison() into compare\_activations()
* Layer dimensions infereed from dimensions dict
* Fix sinabs.network.Network.plot\_comparison() not work correctly for nested ANN and make it only plot Spiking layers' comparison-plot
* updated memory summary to take chip constraints
* samna warning message raised
* open device checks if the device is already open
* moved monitor to make\_config
* added xytp conversion methods
* Added LIF base class
* added warning for discretization
* added test for auto in make\_config
* Added timestamping and memory\_summary methods
* Bug fix: Padding and stride x, y swapped
* Events to raster marked as NotImplemented
* Time stamped events generated
* Forward method defined on events
* Bug fix: config invalid when network uninitialized (no data passed)
* Sub class for flatten batch/time + separate class for IAF
* added bug fix for str 'speck2devkit'
* Added option to specify which layers to monitor in to method
* to device method implemented
* samna device discovery memory errors fixed
* get\_opened\_devices also returns device\_info object
* added get\_device\_map
* Added device\_list
* Added meta class for IAFLayer #5
* Added method to discover connected devices
* added test
* wip: find/move model to device when to() is called
* Config object conditionally created based on device type
* added further test
* Correct error now raised if spiking layer missing at end of network
* speeds up total power use using the new method
* added total synops counter that doesnt use pandas
* speeds up pandas use in synops count, big advantage
* Added io file
* Raise warning when discretize is True and there is an avgpooling layer
* Name convert\_torch\_ann
* necessary change in notebook
* updated docs
* added docs
* Revert "need to test if samna is there"
* bug fixed
* synopcounter tests, changed network and from\_torch accordingly
* moved counter function
* SNNSynopCounter class
* Fix from\_torch method when model contains a ReLU that is not child of a Sequential
* m2r changed to m2r2
* swapped dimensions with batch, default batch None
* membrane reset now implemented properly
* Documentation added and method name renamed to normalize\_weights
* Smart weight rescaling added
* pypi deploy new line added
* sphinx requirements added
* typo in conf.py fixed
* docs folder relocated
* setuptools based setup file
* pbr based project versioning and gitlab ci added
* Samna requirement updated
* fixed cuda issues on from torch, added test
* Method parameter in test corrected
* changed speck to dynapcnn
* fixed mapping problem in auto layer order
* Replace all references to speck as DYNAPCNN, including internal variables
* Type annotation fixed
* Refractored code to dynapcnn from speck
* Changed aiCTX references to SynSense
* fixed bug in discretization of membrane\_subtract (double multiplication)
* membrane reset implementation, removed layer name
* Equation rendering in docs fixed
* Doc front page changed to README
* Added documentation pipeline for testing doc generation
* Setup tools integration for sphinx documentation
* Martino added to authors
* Theme changed to rtd
* added a detach() call
* changed network removing no\_grad
* updated tests to reflect changes in sinabs network
* working bptt notebook
* twine deploy conditional on env variable
* Added condition on env variable to pypi\_deploy
* Add another pipeline that shouldn't execute
* WIP bptt notebook
* CI Lint corrections
* Added test for CI pipeline
* Link to contributing.md file fixed
* Description file content type updated
* Description file content type updated
* Update description type to markdown
* Update development status
* Updated Classifiers
* fixed docs, removed commented-out areas
* removed dependency on samna for validation, and on SpikingLayerBPTT

## v0.2.0

* Threshold gradient scaled by threshold (Bug fix)
* updated docs, removed exclude\_negative\_spikes from fromtorch (no effect)
* test requirements separated
* added coverage
* temporary solution for onnx
* temporary solution for onnxruntime
* amended test requirements to include onnxruntime
* trying to trigger CI
* Updated MNIST notebook
* Instructions for testing added
* \_\_version\_\_ specified from pbr
* Cleaned up setup.py and requirements with pbr
* added coverage tools
* removed network utilities not needed
* updated tests using pathlib
* added some network tests
* WIP on functional docstrings
* removed old stuff from network summary
* update gitignore
* notebook docs updated (WIP)
* fix docs for input shape in from\_torch, removed depency of Network on legacy layers
* removed deprecated arguments of from\_torch
* cleaned up keras in docs
* removed input shape from spiking which caused bugs, and output\_shape from inputlayer
* Changed 'input\_layer' management for sinabs changes'
* change dummy input to device, calculate layer-wise output size
* Updated URL
* Keras-related stuff all removed
* removed pandas from layers
* removed and updated keras tests
* removed summary; device not determined automatically in from\_torch
* removed old tests
* Fixed relative imports
* Added deprecation warning
* Moved layers around, added deprecation
* Moved neuromorphicrelu, quantize, sumpool to separate files, functions to functional
* fixed tests, one not passing
* started changing dvs\_input default
* added dropout
* Unit test for adding spiking output in 'from\_model'
* Enable adding spiking layer to sequential model in from\_torch function
* Roll back changes from last commit and only make sure that meaningful error is produced when last layer is not spiking. Handling of last layer done in sinabs from\_model
* wip: handle networks that end with linear or conv layer
* fixed true\_divide torch fussiness
* removed print statement
* merged commit with sumpool support
* implemented support for sumpool in input network
* Disable default monitor and support one dvs input channel
* Version bump
* removed bad choice
* removed unnecessary calls to print
* fixed bug in old version
* In-code docs for test\_discretized
* Smaller fixes in discretize
* Tests for discretization module
* Added leak management, and test
* individual\_tests made deterministic
* fixed input tests
* valid\_mapping complies with variable naming convention. Extended in-code documentation
* Minor fix in test\_dvs\_input
* Ignore jupyter checkpoints
* Placeholder in tutorial for validation and upload to Speck
* Fixes in test\_dvs\_input
* Rename test\_dvs to test\_dvs\_input
* test\_dvs: Tests with input\_layers
* Warn if both input\_shape and input layer are provided and shapes don't match
* test\_dvs: make sure that missing input specifications are detected
* test made deterministic
* Removed requirement of samna, particularly for tests
* added skip tests with no samna
* doorbell test fixed
* updated large net test to an actual test
* added tests; added support for 3d states
* fixed bug DVS input size
* extended tests to config
* and again
* More updates to deepcopy
* Second deepcopy argument
* Added tentative deepcopy
* deal with missing neuron states
* automatic choice of layer ordering
* add handling swapping layers while searching for a solution
* removed prints, fixed test
* Many fixes needed for the configuration to be valid. Now works
* Documentation for discretize
* Cannot change conv and spk layers, but access them through property. Pool can be changed
* Cannot change conv and spk layers, but access them through property. Pool can be changed
* getting closer
* improvements
* working check on real samna
* validation thing to be compared across machines
* Specklayer correctly handles changing layers. Todo: Update unit tests
* wip: specklayer: make sure that when changing layers, config dict gets updated. TODO: unit test fails
* Property-like behavior for conv/pool/spk layers
* Comparison with original snn only when not discretizing
* Ensure no overwrite of the conv layer during batchnorm merging
* Making sure discretization happens after scaling
* Tutorial for converting from torch model to speck config
* Update documentation
* WIP: Documentation for specklayer. Numpy style docstrings
* WIP: Sphinx documentation
* Minor fixes. Still to do: Discretization of snn (discretize\_sl) does not work)
* Minor fixes in tests
* added ugly workaround to samna-torch crash problem
* fixed bug in sumpool config
* Fixed SumPool
* Completed name change and move of files
* Fix module naming
* deleted references to sumpool2dlayer, loaded sinabs sumpool
* removed unused imports
* uses SumPool from sinabs
* moved test
* updated tests to new locations; new constructor in SpeckNetwork
* moved tests to folder
* deleted scratch folder
* Tests related to dvs
* Fixes wrt to handling dvs and pooling, completed type hints
* wrote docstrings
* should now be safe to commit init
* some minor changes
* added test, changed var names
* small correction to previous commit
* added support for a specific case of batchnorm
* Use deepcopy for copying layers
* merge bc of black
* Avg pooling now turned to sum pooling and weights rescaling (1 failing test)
* Test to verify that all layers are copy and not references
* Make sure all layers in SpeckCompatibleNetwork are copies of the original
* (WIP) started implementing transfer to sumpool
* Workaround for copying spiking layers in discretize\_conv\_spike
* updated and added tests
* fixed several issues that arose with testing
* bugfix: reset\_states in network
* correct way of ignoring neurons states
* discretization now optional (for testing)
* input shape removed where not needed; more cleanup
* Minor
* separated make\_config from the rest
* a little cleanup and commenting
* seemingly working class-based version
* somewhat working version of class-based
* Handle Linear layers and Flatten, ignore Dropout2d
* started transformation into class
* added gitignore
* updated new api of samna
* added smartdoor test
* Doorbell test
* Un-comment speck related lines
* minor
* samna independent test-mode for fixing some bugs
* Fixing bugs
* Wip: update for sinabs 0.2 - discretization
* Wip: update tospeck for compatibility for sinabs 0.2
* Wip: update tospeck for compatibility for sinabs 0.2
* Refactored keras\_model -> analog\_model
* Added tool to compute output shapes
* correct device for spiking layers
* added tentative synops support
* version number updated
* updated file paths in tests
* threshold methods updated, onnnx conversion works now
* wip:added test for equivalence
* fixed bug from\_torch was doing nothing
* model build method separately added
* changed default membrane subtract to Threshold, as in IAF. implemented in from\_torch
* updated documentation
* fixed bug in from\_torch; negative spikes no longer supported
* onnx support for threshold operation
* updated test; removed dummy input shape
* added warnings for unsupported operations
* Input shape optional and neurons dynamically allocated
* from\_torch completely rewritten (WIP)
* wip: from\_torch refractoring
* marked all torch layer wrappers to deprecated
* Depricated TorchLayer added
* merged master to bptt\_devel

## v0.1.dev7 (09/04/2020)

* install m2r with hotfix for new version of sphinx
* changed membrane\_subtract and reset defaults, some cleanup
* added test to compare iaf implementations
* added dummy test file intended for bptt layer
* removed content from init file, since it breaks for people who do not have dependencies
* sumpool layer stride is None bug
* introduced test of discretization in simulation
* Made necessary changes to actually simulate the discretized network
* added bias check in descretize sc2d
* merged changes from features/separate\_discretization
* bugfixes
* fix import
* misc
* Fix bias shapes and handling of 'None'-states
* merge updates from feature/spiking\_model\_to\_speck
* Fix biases shape
* wip
* updated version number
* added support for batch mode operation
* Fixes in neuron states and weight shapes. Updated test
* Undo reverting commit 5af49846 and fix dimensions for neuron states
* Fix weight dimensions
* added conversion of flatten and linear to convolutions
* Use speckdemo module and handle models without biases
* provided default implementation of iaf\_bptt to passthrough
* Small fix in plotting in test
* Improved in-code documentation of tospeck.py
* Test script for porting a simple spiking model to a speck config
* Quantization of weights, biases and thresholds
* SpikingLayer with learning added to layers without default import
* Bugfixes in tospeck.py
* can handle sequential models of SpikingConv2dLayers and SumPooling2dLayers
* Remove tests that should be handled by ctxctl
* wip: handling of pooling layers
* For compatibility issues that result in not matching dimensions raise exceptions instead of warnings
* WIP: Method for converting Spiking Model to speck configurations
* SpikingLayer attributes membrane\_subtract and membrane\_reset as properties to avoid that both are None at the same time
* WIP: Method for converting SpikingConv2DLayer to speck configurations
* threshold function fix, bptt mnist example with spiking layer in notebook
* threshold functions used in forward iaf method for bptt
* added differentiable threshold functions
* bugfix related to sumpool
* added synopscount to master
* added documentation synopcounter and sumpool
* added new layers to docs
* Added analogue sumpool layer
* added two  layers by qian
* updated summary function for iaf\_tc
* added synoploss and refactored
* added classifiers to setup.py
* fixed typos in setup.py
* updated setup file
* updated branch to master for pypi deployment
* fixed reference to rockpool in tag
* upload to pypi and tags in readme file
* version bump for test
* direct execute with twine
* typo fix
* added tags of the runner
* pypi build triggered on pip branch
* removed trailing line
* ci script to upload to test pypi
* wip: adding pip support for sinabs
* added option to only reproduce current instead of spikes
* added clear buffer method
* pew workon to pew ls
* added pew to documentation
* round on stochastic rounding eval mode
* stochastic rounding only during learning
* added stochastic rounding option to NeuromorphicReLU
* added normalization level to sig2spike layer
* updated documentation structure and pipenv tutorial
* modified iaf tc's expecte dims to be [t, ch]
* merged changes from master
* fixed missing module sinabs.from\_keras
* fixed tensorflow version 1.15
* fixed tensorflow version in ci script
* added tensorflow install to ci script
* typo fix
* force install torch
* updated documentation for from\_keras
* added pipfile
* moved all from keras methods to from\_keras.py
* added doc string
* added rescaling of biases to from\_torch
* breaking change to Sig2SpikeLayer
* time steps computed based on dimensions of synaptic output
* functioning code for spiking model
* renamed TorchLayer to Layer, TDS to TemporalConv1d
* added kernel shape for tds layer
* fixed cuda imports in tests
* merged master
* updated notebook with a full run time output
* added mnist\_cnn weight matrix for the examples to run smoothly
* example of from\_torch Lenet 5
* example of from\_torch Lenet 5
* lenet example from\_torch, and in Chinese
* missing import added
* quantize layers are not called by name any more
* supported avgpool with different kernel sizes
* added some documentation, quantize now does nothing
* fix linear layer and add sumpool layer to from\_torch
* clean up maxpooling2d
* clean up maxpoooling2d
* fix maxpooling spike\_count error
* fix maxpooling spike\_count error
* initial mock code
* implemented quantization
* Initial commit
* load DynapSumPool and DynapConv2dSynop from pytorch
* added flag to exclude negative spikes
* added support for neuromorphicrelu
* updated setup file to specify tensorflow version dependency
* Some minor changes
* fixes summary
* threshold management in from torch
* functionalities added to torch vonverter
* line-height fixed in h1
* added intro to snns notebook documentation
* merge errors fixed in init file
* synops to cpu
* fixes needed for summary and synops
* merged init file
* init file merged
* overwrote forward method
* removed detach()
* all self.spikes\_number are numbers only and detached now
* fixed incorrect variable name for weights
* added SpikingLinearLayer
* doc string corrections
* fixed test following small refactor
* fixed documentation
* allowed threshold\_low setting
* added documentation
* added YOLO layer and converted converter
* converter uses Sequential and Network instead of ModuleList
* merged latest version (PR) or no\_spike\_tracking
* iaf layers do not save spikes
* fixed loss return with flag
* trivial merge of no spike tracking
* removed status caching and sum(0)
* merged no spike tracking but test not fixed
* iaf layers do not save spikes
* changed copying strategy to avoid warnings
* Sadique worked on clearing cache on iaf forward()
* img\_to\_spk fix
* small improvements to spkconverter
* linear imgtospk
* spike converter from torch and test
* remove unwanted prints
* small changes useful for yolo
* implemented linear mode for conv2d
* changes to synaptic output
* implemented spike generation layer from analog signals
* fixed causal convolutions and padding
* implemented delay buffer
* added initial code for time delayed spiking layer
* added image to spike conversion layer
* added conv1d to the documentation
* added conv1d layer
* updated notebooks in examples
* conversion from markdown fixed
* added link to gitlab pages in readme
* documentation added to pages
* updated branch for testing and building
* fixed path to build folder
* pip upgrade command missing pip
* added gitlab CI script
* state to cuda device
* license notice updated in setup file
* layers submodule added to setupfile
* fixed calls to np load with allow\_pickle arg
* added conv3d layer
* initial code
* merged
* fixed typos in readme

## v0.1.0

* fixed version number
* removed contrib branch
* added initial text for contributions file
* updated mirror url
* Added contact text
* added contributing file
* added license text to readme
* added AGPL license notice to all files in library
* added LeNet 5 example
* default neuron parameters updated to work out of the box
* added convtranspose2d layer
* abstract class SpikingLayer added to documentation
* iaf code moved to abstract class
* summary added to layer base class
* summary modified
* update example to generate and readout spike trains
* max pooling keras
* restored readme text
* added readme in docs folder
* added license AGPL
* auto rescale multiple average pooling in row
* fix quantizing nBits for weights and threshold
* softmax means ReLU for inference and fix auto-rescaling
* push test
* push test
* summary modified
* added build to gitignore list
* typos in readme
* updated documentation file structure
* Initial file commit
* Initial commit
