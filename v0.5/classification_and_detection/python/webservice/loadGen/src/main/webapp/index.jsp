<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Insert title here</title>
</head>
<link rel="stylesheet" type="text/css" href="statics/css/global.css"/>

<body>
welcome to the load generator page!
<br>follow the links below to config and test

<br><a href="goOnlineQuery.do?serviceId=0" target="_blank">turn to the GUI page, please wait a few time</a>
<br><a href="startOnlineQuery.do?intensity=1&serviceId=0" target="_blank">start the load generator</a>
<br><a href="stopOnlineQuery.do?serviceId=0" target="_blank">stop the load generator</a>
<br><a href="setIntensity.do?intensity=10&serviceId=0" target="_blank">dynamically change the request number per second</a>
  
</body>
</html>