# 09 FoundryIQ — no module Data/

The reference repo `deployed-in-azure/RAG/09_FoundryIQ` has no `Data/` folder.
FoundryIQ queries Azure AI Search Knowledge Bases (Contoso cloud / health-plans /
job-roles knowledge sources in the sample) rather than shipping synthetic starship
documents. Phase 8 will wire against a provisioned Knowledge Base; there is no
`index_definition.json` to port for this module.
