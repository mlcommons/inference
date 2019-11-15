<%@ page language="java" import="java.util.*" pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%
String path = request.getContextPath();
String basePath = request.getScheme()+"://"+request.getServerName()+":"+request.getServerPort()+path+"/";
%>

<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
 
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="viewport"
	content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
<meta name="renderer" content="webkit">
<title>Load generator</title>
<link rel="stylesheet" href="statics/css/pintuer.css">
<link rel="stylesheet" href="statics/css/admin.css">
<script src="statics/js/jquery.js"></script>
<script src="statics/js/pintuer.js"></script>
</head>
<body style="height: 3300px;">
	<div> 
		<div id="chart">
			<div id="web" style="width: 1250px; height: 350px; position: absolute; left: 50px; top: 0px;"></div>
	 	</div>
		<div id="AvgDiv" style="width: 50px; height: 50px; position: absolute; left: 1150px; top: 13px;">Avg99th:<span id="avg"></span>ms</div>
		<div id="QpsDiv" style="width: 150px; height: 50px; position: absolute; left: 1140px; top: 360px;">AvgRPS:<span id="avg_rps"></span>&nbsp;&nbsp;AvgQPS:<span id="avg_qps"></span>&nbsp;&nbsp;SR:<span id="serviceRate"></span>%</div>
		<div id="QpsDiv" style="width: 150px; height: 50px; position: absolute; left: 1140px; top: 380px;">realRPS:<span id="real_rps"></span>&nbsp;&nbsp;realQPS:<span id="real_qps"></span></div>
	</div>
	<script type="text/javascript" src="statics/js/jquery-1.9.1.js"></script>
	<script type="text/javascript" src="statics/js/highcharts.js"></script>
	<script type="text/javascript" src="statics/js/highcharts-more.js"></script>
	<script type="text/javascript" src="statics/js/highstock.js"></script>
	<script type="text/javascript" src="statics/js/exporting.js"></script>
	<script type="text/javascript" src="statics/js/highcharts-zh_CN.js"></script>
	<script type="text/javascript">
Highcharts.setOptions({ 
	global: { 
		useUTC: false 
		} 
	});
Highstock.setOptions({ 
	global: { 
		useUTC: false 
		} 
    });
    var lastcollecttime=null;
    var elementAvgQps=document.getElementById('avg_qps');
    var elementAvgRps=document.getElementById('avg_rps');
    var elementRealQps=document.getElementById('real_qps');
    var elementRealRps=document.getElementById('real_rps');
    var elementSR=document.getElementById('serviceRate');
    var elementAvg=document.getElementById('avg');
$(document).ready(function() {
	Highcharts.chart('web',{
        chart: {
            type: 'line',//scatter
            zoomType: 'x',
            events: {
                load: function (){
                    var series = this.series[0]; 
                    var x,queryTime,real_qps,real_rps,avg_rps,avg_qps,avg,serviceRate;
                    var serviceId=${serviceId};
                    setInterval(function (){
                    	$.ajax({
            				async:true,
            				type:"get",
            				url:"getOnlineQueryTime.do",
            				data:{serviceId:serviceId},
        					dataType:"json",
            				success:function(returned){
            					if(returned!=null&&returned!=""){
            						x = returned[0].generateTime;
            						queryTime = returned[0].queryTime99th;
    							    avg_qps = returned[0].avgQps;
    							    avg_rps = returned[0].avgRps;
    							    real_qps = returned[0].realQps;
    							    real_rps = returned[0].realRps;
    							    serviceRate = returned[0].windowAvgServiceRate;
    							    avg = returned[0].OnlineAvgQueryTime;
            						if(lastcollecttime==null){//如果第一次判断 直接添加点进去
      			            	    	 series.addPoint([x,queryTime], true, true); 
      			            	    	 elementAvgRps.innerHTML=avg_rps;
      			            	    	 elementAvgQps.innerHTML=avg_qps;
      			            	    	 elementRealRps.innerHTML=real_rps;
     			            	    	 elementRealQps.innerHTML=real_qps;
      			            	    	 elementSR.innerHTML=serviceRate;
      			            	    	 elementAvg.innerHTML=avg;
      			            	    	 lastcollecttime = x;
      			            	    }else{ 
      			            	    	if(lastcollecttime<x){//如果不是第一次判断，则只有上次时间小于当前时间时才添加点
      			            	    		series.addPoint([x,queryTime], true, true); 
      			            	    		elementAvgRps.innerHTML=avg_rps;
         			            	    	elementAvgQps.innerHTML=avg_qps;
         			            	    	elementRealRps.innerHTML=real_rps;
        			            	    	elementRealQps.innerHTML=real_qps;
         			            	    	elementSR.innerHTML=serviceRate;
         			            	    	elementAvg.innerHTML=avg;
         			            	    	lastcollecttime = x;
      			            	    	}
      			            	    } 
            					}
            				}	
            			}); 
                    }, 1000);
                }
            }
        },
        plotOptions: {
            series: {
                marker: {
                    radius: 2
                }
            }
        },
        boost: {
            useGPUTranslations: true
        },
        xAxis: {
            type: 'datetime',
            tickPixelInterval: 150
            
        },
        title: {
            text: 'online query latency'
        },
        legend: {                                                                    
            enabled: false                                                           
        } ,
        yAxis: {
            title: {
                text: '99thLatencyPerSecond/ms'
            },
        },
        tooltip: {
            formatter:function(){
                return'<strong>'+this.series.name+'</strong><br/>'+
                    Highcharts.dateFormat('%Y-%m-%d %H:%M:%S.%L',this.x)+'<br/>'+'99thLatencyPerSecond：'+this.y+' ms';
            },
        },
        series: [${seriesStr}]
    });
	
   
});
</script>

</body>
</html>
