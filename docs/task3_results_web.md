<!-- 
working file, when finished copy to the website -> .../BioDCASE-Website/content/pages/challenge2026/task3.results.md
-->

Title: Bioacoustics for Tiny Hardware
Date: 2026-05-18 18:15
Category: challenge
Slug: challenge2026/task3-results
Author: DCASE
HeaderTextSecondary: Challenge results
HeaderImage: images/header/veriko-dundua-BPKTTohN8wU-unsplash.jpg
HeaderImagecc: Background photo by Veriko Dundua
btoc: True
btoc_panel_color: panel-default
Template: page_wide


# About
This is the result page of the BioDCASE Challenge 2026.
More details can be found on the task page:

<a href="{filename}/pages/challenge2026/task3.md" class="btn btn-primary" style="">Task description page</a>


# Honors
<!-- 
todo: make this a bit prettier
-->
<div class="award-block award-block-winning">
    <div class="award-block-title">Winning Submission</div>
    <div class="award-block-score">The highest ranking submission, achieving an achieving an rank score of xxx is:</div>
    <div class="award-block-authors">name1 lastname1 • name1 lastname2...</div>
    <div class="award-block-institution">institution</div>
    <div class="award-block-system">technical report title</div>
    <div class="award-block-institution">BioDCASE 2026, Task 3</div>
</div>

<div class="award-block award-block-jury">
    <div class="award-block-title">Jury Award</div>
    <div class="award-block-authors">name1 lastname1 • name1 lastname2...</div>
    <div class="award-block-institution">institution</div>
    <div class="award-block-system">technical report title</div>
    <div class="award-block-comment">our comment</div> 
    <div class="award-block-institution">BioDCASE 2026, Task 3</div>
</div>


# Teams ranking
<table class="datatable table table-hover table-condensed"
  data-yaml="content/data/challenge2026/results_task3_entries.yaml"
  data-id-field="code"
  data-sort-name="rank_team"
  data-sort-order="asc"
  data-pagination="true"
  data-show-pagination-switch="true"
  data-show-rank="false"
  data-rank-mode="grouped_muted"
  data-show-chart="false"
  data-chart-modes="bar,scatter"
  data-chart-default-mode="scatter"
  data-row-highlighting="true"
  data-bar-hline="true"
  data-bar-show-xaxis="false"
  data-bar-tooltip-position="top"
  data-bar-show-vertical-indicator="true"
  data-bar-vertical-indicator-linewidth="2"
  data-bar-show-horizontal-indicator="true"
  data-bar-horizontal-indicator-linewidth="2"
  data-scatter-x="rank_team"
  data-scatter-y="rank_score">
<!--   
  table construction:
  -
    rank
  submission information
    name
    technical report
  rank
    official rank
    rank score

  --  
  classification performance
    acc infer
    acc tflite
    roc-auc infer
    roc-auc tflite
  resource efficiency
    model size inf. [kB]
    model size tfl. [kB]
    MACS inf. [M ops]
    MACS tfl. [M ops]
  - or
  inference model
    acc
    roc-auc
    model size [kB]
    MACs [M ops.]
  embedding model (.tflite)
    acc
    roc-auc
    model size [kB]
    MACs [M ops.]
  --

  embedding performance
    RAM Usage [kB]
    Time Setup [ms]
    Time Preproc. [ms]
    Time Model [ms]
    Time Total [ms]
  -
    rank score
 -->
 <!-- 
  scores:
  accuracy_inference: 0.7273
  accuracy_tflite: 0.7273
  roc_auc_inference: 0.9636
  roc_auc_tflite: 0.9545
  model_size_inference_bytes: 393237
  model_size_tflite_bytes: 111408
  macs_inference: 23326923
  macs_tflite: 23315232
  num_params_inference: 97163
  num_params_tflite: 96896
  embedded_time_ms_setup: N/A
  embedded_time_ms_preprocessing: N/A
  embedded_time_ms_model: N/A
  embedded_time_ms_total: N/A
  embedded_ram_usage_bytes: N/A 
  -->
  <thead>
    <tr>
      <!-- <th></th> -->
      <th class="sep-left-cell" colspan="1"></th>
      <th class="sep-left-cell text-center" colspan="2">Submission information</th>
      <!-- <th class="sep-left-cell text-center" colspan="2">Ranking</th> -->
      <th class="sep-left-cell text-center" colspan="4">Inference Model</th>
      <th class="sep-left-cell text-center" colspan="4">Embedding Model (.tflite)</th>
      <th class="sep-left-cell text-center" colspan="5">Embedded Performance on ESP32-S3)</th>
      <th class="sep-left-cell" colspan="1"></th>
      <!--       
      <th class="sep-left-cell text-center" colspan="4">Classification Performance</th>
      <th class="sep-left-cell text-center" colspan="4">Resource Efficiency</th>
      <th class="sep-left-cell text-center" colspan="5">Embedding Performance</th>
      -->
    </tr>
    <tr>
      <!-- <th class="sep-right-cell" data-rank="true">Rank</th> -->
      <!-- rank -->
      <th class="sep-left-cell text-center" 
          data-field="rank_team" 
          data-sortable="true" 
          data-chartable="true"
          data-value-type="int"
          data-axis-label="Team rank">
          Rank
      </th>
      <!-- submission information -->
      <th class="sm-cell"
        data-field="name"
        data-sortable="true">
        Name
      </th>
      <th class="text-center" 
        data-field="bibtex_key" 
        data-sortable="false"
        data-value-type="anchor">
        Technical<br></br>Report
      </th>
      <!-- inference model -->
      <th class="text-center" 
        data-field="accuracy_inference" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float4"
        data-axis-label="Acc. inf.">
        Acc.
      </th>
      <th class="text-center" 
        data-field="roc_auc_inference" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float4"
        data-axis-label="ROC-AUC inf.">
        ROC-AUC
      </th>
      <th class="text-center" 
        data-field="model_size_inference_bytes" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float3"
        data-axis-label="Model Size inf. [Bytes]">
        Model Size [Bytes]
      </th>
      <th class="text-center" 
        data-field="macs_inference" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float3"
        data-axis-label="MACs inf.">
        MACs
      </th>
      <!-- embedding model -->
      <th class="text-center" 
        data-field="accuracy_inference" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float4"
        data-axis-label="Acc. tfl.">
        Acc.
      </th>
      <th class="text-center" 
        data-field="roc_auc_inference" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float4"
        data-axis-label="ROC-AUC tfl.">
        ROC-AUC
      </th>
      <th class="text-center" 
        data-field="model_size_inference_bytes" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float3"
        data-axis-label="Model Size tfl. [Bytes]">
        Model Size [Bytes]
      </th>
      <th class="text-center" 
        data-field="macs_inference" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float3"
        data-axis-label="MACs inf.">
        MACs
      </th>
      <!-- Embedded Performance on ESP32-S3 -->
      <th class="text-center" 
        data-field="embedded_ram_usage_bytes" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float3"
        data-axis-label="RAM Usage [Bytes]">
        RAM Usage [Bytes]
      </th>
      <th class="text-center" 
        data-field="embedded_time_ms_setup" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float3"
        data-axis-label="Time Setup [ms]">
        Time Setup [ms]
      </th>
      <th class="text-center" 
        data-field="embedded_time_ms_preprocessing" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float3"
        data-axis-label="Time Preproc. [ms]">
        Time Preproc. [ms]
      </th>
      <th class="text-center" 
        data-field="embedded_time_ms_model" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float3"
        data-axis-label="Time Model [ms]">
        Time Model [ms]
      </th>
      <th class="text-center" 
        data-field="embedded_time_ms_total" 
        data-sortable="true" 
        data-chartable="true"
        data-value-type="float3"
        data-axis-label="Time Total [ms]">
        Time Total [ms]
      </th>
      <!-- rank score -->
      <th class="sep-left-cell text-center"
        data-field="rank_score"
        data-sortable="true"
        data-chartable="true"
        data-value-type="float2"
        data-axis-label="Rank Score">
        Rank Score
      </th>
    </tr>
  </thead>
</table>


# Technical reports

<div class="btex" data-source="content/data/challenge2026/technical_reports_task3.bib" data-stats="true">
  <div class="panel-group" id="accordion" role="tablist" aria-multiselectable="true">
    <div class="panel-group" id="accordion" role="tablist" aria-multiselectable="true">
      {% for item in publications|sort(attribute='key') %}
        <div class="panel publication-item" id="{{ item.key }}" style="box-shadow: none">
          <div class="panel-heading" role="tab" id="heading-{{ item.key }}">
            <div class="row">
              <div class="col-xs-9">
                <h4>{{item.title | replace("{","") | replace("}","") }}</h4>
                <p style="text-align:left">{{item._authors}}</p>
                <p style="text-align:left"><em>{{item._affiliations}}</em></p>
                <p style="text-align:left">{{item._extra_info}}</p>
                <p style="text-align:left">
                {% if item.award %}<span class="label label-success">{{item.award}}</span>{% endif %}
                {% if item.cites %}
                <span style="padding-left:5px">
                <span title="Number of citations" class="badge">{{ item.cites }} {% if item.cites==1 %}cite{% else %}cites{% endif %}</span>
                </span>
                {% endif %}
                </p>
                <button type="button" class="btn btn-default btn-xs" data-toggle="collapse" data-parent="#accordion" href="#collapse-{{ item.key }}" aria-expanded="true" aria-controls="collapse-{{ item.key }}">
                <i class="fa fa-caret-down"></i> Details...</button>
              </div>
              <div class="col-xs-3">
                <div class="btn-group">
                  <button type="button" class="btn btn-xs btn-danger" data-toggle="modal" data-target="#bibtex-{{ item.key }}"><i class="fa fa-file-text-o"></i> Bibtex</button>
                  {% if item.pdf %}
                    <a href="{{item.pdf}}" class="btn btn-xs btn-warning" style="text-decoration:none;border-bottom:0;padding-bottom:2px" rel="tooltip" title="Download pdf" data-placement="bottom"><i class="fa fa-file-pdf-o fa-1x"></i> PDF</a>
                  {% endif %}
                  {% if item.demo %}
                    <a href="{{item.demo}}" class="btn btn-xs btn-primary iframeDemo" style="text-decoration:none;border-bottom:0;padding-bottom:2px" rel="tooltip" title="Demo" data-placement="bottom"><i class="fa fa-headphones"></i> Demo</a>
                  {% endif %}
                  {% if item.toolbox %}
                    <a href="{{item.toolbox}}" class="btn btn-xs btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:2px" rel="tooltip" title="Toolbox" data-placement="bottom"><i class="fa fa-file-code-o"></i> Toolbox</a>
                  {% endif %}
                  {% if item.data1 or item.data2 %}
                    <a href="" onclick="$('#collapse-{{ item.key }}').collapse('show');window.location.hash='#{{ item.key }}';return false;" class="btn btn-xs btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:2px" rel="tooltip" title="{{item.code1.title}}" data-placement="bottom"><i class="fa fa-database"></i> Dataset</a>
                  {% endif %}
                  {% if item.code1 or item.code2 or item.code3 or item.code4 or item.code5 %}
                    <a href="" onclick="$('#collapse-{{ item.key }}').collapse('show');window.location.hash='#{{ item.key }}';return false;" class="btn btn-xs btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:2px" rel="tooltip" title="{{item.code1.title}}" data-placement="bottom"><i class="fa fa-file-code-o"></i> Code</a>
                  {% endif %}
                  {% if item.git1 or item.git2 or item.git3 or item.git4 %}
                    <button type="button" class="btn btn-xs btn-success" data-toggle="collapse" data-parent="#accordion" href="#collapse-{{ item.key }}" aria-expanded="true" aria-controls="collapse-{{ item.key }}">
                      <i class="fa fa-git"></i> Code
                    </button>
                  {% endif %}
                </div>
              </div>
            </div>
          </div>
          <div id="collapse-{{ item.key }}" class="panel-collapse collapse" role="tabpanel" aria-labelledby="heading-{{ item.key }}">
            <div class="panel-body well well-sm">
              <h4>{{item.title | replace("{","") | replace("}","") }}</h4>
              <p style="text-align:left"><small>{{item._authors}}</small><br><small><em>{{item._affiliations}}</em></small></p>
              {% if item.abstract %}
                <h5><strong>Abstract</strong></h5>
                <p class="text-justify">{{item.abstract}}</p>
              {% endif %}
              {% if item.keywords %}
                <h5><strong>Keywords</strong></h5>
                <p class="text-justify">{{item.keywords}}</p>
              {% endif %}
              {% if item.award %}
                <p><strong>Awards:</strong> {{item.award}}</p>
              {% endif %}
              {% if item._system_input or item._system_sampling_rate or item._system_data_augmentation or item._system_features or item._system_classifier or item._system_decision_making %}
                <h5><strong>System characteristics</strong></h5>
                <table class="table table-condensed">
                {% if item._system_input %}
                <tr><td class="col-md-3">Input</td><td>{{item._system_input}}</td></tr>
                {% endif %}
                {% if item._system_sampling_rate %}
                <tr><td class="col-md-3">Sampling rate</td><td>{{item._system_sampling_rate}}</td></tr>
                {% endif %}
                {% if item._system_data_augmentation %}
                <tr><td class="col-md-3">Data augmentation</td><td>{{item._system_data_augmentation}}</td></tr>
                {% endif %}
                {% if item._system_features %}
                <tr><td class="col-md-3">Features</td><td>{{item._system_features}}</td></tr>
                {% endif %}
                {% if item._system_embeddings %}
                <tr><td class="col-md-3">Embeddings</td><td>{{item._system_embeddings}}</td></tr>
                {% endif %}                                
                {% if item._system_classifier %}
                <tr><td class="col-md-3">Classifier</td><td>{{item._system_classifier}}</td></tr>
                {% endif %}
                {% if item._system_decision_making %}
                <tr><td class="col-md-3">Decision making</td><td>{{item._system_decision_making}}</td></tr>
                {% endif %}
                {% if item._system_complexity_management %}
                <tr><td class="col-md-3">Complexity management</td><td>{{item._system_complexity_management}}</td></tr>
                {% endif %}
                {% if item._system_pipeline %}
                <tr><td class="col-md-3">Pipeline</td><td>{{item._system_pipeline}}</td></tr>
                {% endif %}
                {% if item._system_framework %}
                <tr><td class="col-md-3">Framework</td><td>{{item._system_framework}}</td></tr>
                {% endif %}
                </table>
              {% endif %}
              {% if item.cites %}
                <p><strong>Cites:</strong> {{item.cites}} (<a href="{{ item.citation_url }}" target="_blank">see at Google Scholar</a>)</p>
              {% endif %}
              <div class="btn-group">
                <button type="button" class="btn btn-sm btn-danger" data-toggle="modal" data-target="#bibtex-{{ item.key }}"><i class="fa fa-file-text-o"></i> Bibtex</button>
                {% if item.pdf %}
                  <a href="{{item.pdf}}" class="btn btn-sm btn-warning" style="text-decoration:none;border-bottom:0;padding-bottom:6px" rel="tooltip" title="Download pdf" data-placement="bottom"><i class="fa fa-file-pdf-o fa-1x"></i> PDF</a>
                {% endif %}
                {% if item.slides %}
                  <a href="{{item.slides}}" class="btn btn-sm btn-info" style="text-decoration:none;border-bottom:0;padding-bottom:6px" rel="tooltip" title="Download slides" data-placement="bottom"><i class="fa fa-file-powerpoint-o"></i> Slides</a>
                {% endif %}
                {% if item.webpublication %}
                  <a href="{{item.webpublication.url}}" class="btn btn-sm btn-info" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.webpublication.title}}"><i class="fa fa-book"></i> Web publication</a>
                {% endif %}
              </div>
              <div class="btn-group">
              {% if item.toolbox %}
                <a href="{{item.toolbox}}" target="_blank" class="btn btn-sm btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:6px" rel="tooltip" title="Toolbox" data-placement="bottom"><i class="fa fa-file-code-o"></i> Toolbox</a>
              {% endif %}
              {% if item.data1 %}
                <a href="{{item.data1.url}}" target="_blank" class="btn btn-sm btn-info" style="text-decoration:none;border-bottom:0;padding-bottom:6px" rel="tooltip" title="Toolbox" data-placement="bottom"><i class="fa fa-database"></i> {{item.data1.title}}</a>
              {% endif %}
              {% if item.data2 %}
                <a href="{{item.data2.url}}" target="_blank" class="btn btn-sm btn-info" style="text-decoration:none;border-bottom:0;padding-bottom:6px" rel="tooltip" title="Toolbox" data-placement="bottom"><i class="fa fa-database"></i> {{item.data2.title}}</a>
              {% endif %}
              {% if item.code1 %}
                <a href="{{item.code1.url}}" target="_blank" class="btn btn-sm btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.code1.title}}"><i class="fa fa-file-code-o"></i> {{item.code1.title}}</a>
              {% endif %}
              {% if item.code2 %}
                <a href="{{item.code2.url}}" target="_blank" class="btn btn-sm btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.code2.title}}"><i class="fa fa-file-code-o"></i> {{item.code2.title}}</a>
              {% endif %}
              {% if item.code3 %}
                <a href="{{item.code3.url}}" target="_blank" class="btn btn-sm btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.code3.title}}"><i class="fa fa-file-code-o"></i> {{item.code3.title}}</a>
              {% endif %}
              {% if item.code4 %}
                <a href="{{item.code4.url}}" target="_blank" class="btn btn-sm btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.code4.title}}"><i class="fa fa-file-code-o"></i> {{item.code4.title}}</a>
              {% endif %}
              {% if item.code5 %}
                <a href="{{item.code4.url}}" target="_blank" class="btn btn-sm btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.code5.title}}"><i class="fa fa-file-code-o"></i> {{item.code5.title}}</a>
              {% endif %}
              {% if item.git1 %}
                <a href="{{item.git1.url}}" target="_blank" class="btn btn-sm btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.git1.title}}"><i class="fa fa-git"></i> {{item.git1.title}}</a>
              {% endif %}
              {% if item.git2 %}
                <a href="{{item.git2.url}}" target="_blank" class="btn btn-sm btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.git2.title}}"><i class="fa fa-git"></i> {{item.git2.title}}</a>
              {% endif %}
              {% if item.git3 %}
                <a href="{{item.git3.url}}" target="_blank" class="btn btn-sm btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.git3.title}}"><i class="fa fa-git"></i> {{item.git3.title}}</a>
              {% endif %}
              {% if item.git4 %}
                <a href="{{item.git4.url}}" target="_blank" class="btn btn-sm btn-success" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.git4.title}}"><i class="fa fa-git"></i> {{item.git4.title}}</a>
              {% endif %}
              {% if item.demo %}
                <a href="{{item.demo}}" target="_blank" class="btn btn-sm btn-primary iframeDemo" style="text-decoration:none;border-bottom:0;padding-bottom:6px" rel="tooltip" title="Demo" data-placement="bottom"><i class="fa fa-headphones"></i> Demo</a>
              {% endif %}
              {% if item.link1 %}
                <a href="{{item.link1.url}}" target="_blank" class="btn btn-sm btn-info" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.link1.title}}"><i class="fa fa-external-link-square"></i> {{item.link1.title}}</a>
              {% endif %}
              {% if item.link2 %}
                <a href="{{item.link2.url}}" target="_blank" class="btn btn-sm btn-info" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.link2.title}}"><i class="fa fa-external-link-square"></i> {{item.link2.title}}</a>
              {% endif %}
              {% if item.link3 %}
                <a href="{{item.link3.url}}" target="_blank" class="btn btn-sm btn-info" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.link3.title}}"><i class="fa fa-external-link-square"></i> {{item.link3.title}}</a>
              {% endif %}
              {% if item.link4 %}
                <a href="{{item.link4.url}}" target="_blank" class="btn btn-sm btn-info" style="text-decoration:none;border-bottom:0;padding-bottom:6px" title="{{item.link4.title}}"><i class="fa fa-external-link-square"></i> {{item.link4.title}}</a>
              {% endif %}
              </div>
            </div>
          </div>
        </div>
        <!-- Modal -->
        <div class="modal fade" id="bibtex-{{ item.key }}" tabindex="-1" role="dialog" aria-labelledby="bibtex-{{ item.key }}label" aria-hidden="true">
          <div class="modal-dialog">
            <div class="modal-content">
              <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal"><span aria-hidden="true">&times;</span><span class="sr-only">Close</span></button>
                <h4 class="modal-title" id="bibtex{{item.key}}label">{{item.title}}</h4>
              </div>
              <div class="modal-body">
                <pre>{{item.bibtex}}</pre>
              </div>
              <div class="modal-footer">
                <button type="button" class="btn btn-default" data-dismiss="modal">Close</button>
              </div>
            </div>
          </div>
        </div>
      {% endfor %}
    </div>
  </div>
</div>
<br>


<script>
(function($) {
  $(document).ready(function() {
    var hash = window.location.hash.substr(1);
    var anchor = window.location.hash;

    var shiftWindow = function() {
      var hash = window.location.hash.substr(1);
      if($('#collapse-'+hash).length){
        scrollBy(0, -100);
      }
    };
    window.addEventListener("hashchange", shiftWindow);

    if (window.location.hash){
      window.scrollTo(0, 0);
      history.replaceState(null, document.title, "#");
      $('#collapse-'+hash).collapse('show');
      setTimeout(function(){
        window.location.hash = anchor;
        shiftWindow();
      }, 2000);
    }
  });
})(jQuery);
</script>